from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agents_llm import KnowledgeAgent, ResolutionAgent, TriageAgent
from .attachments import AttachmentParser, attachment_evidence_to_doc
from .checkpointing import build_checkpointer
from .config import TicketFlowSettings
from .db import TicketFlowRepository
from .governance import build_action_route, get_tool_approval_policy
from .intent import BertIntentRecognizer
from .llm import OpenAICompatClient
from .mcp_email_client import MCPEmailClient
from .models import (
    AttachmentEvidence,
    ActionRoutingDecision,
    ActionProposal,
    ContextSufficiencyResult,
    ExternalOpRecord,
    InvocationResult,
    ReplyFactCheckResult,
    ReviewDecision,
    TicketFlowState,
    TicketRecord,
    TicketResponse,
    ToolApprovalPolicy,
)
from .rag import HybridRetriever
from .seed import build_seed_dataset
from .tools import TicketTools


def _append_trace(state: TicketFlowState, actor: str, step: str, detail: str, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({"actor": actor, "step": step, "detail": detail, "payload": payload or {}})
    return trace


def _append_audit(state: TicketFlowState, tools: TicketTools, ticket_id: str, actor: str, event_type: str, detail: str, payload: dict[str, Any] | None = None) -> list[Any]:
    audit_log = list(state.get("audit_log", []))
    event = tools.make_audit_event(actor, event_type, detail, payload)
    audit_log.append(event)
    tools.save_audit_log(ticket_id, actor, event_type, detail, payload or {})
    return audit_log


@contextmanager
def _env_overrides(values: dict[str, str] | None):
    if not values:
        yield
        return
    old_values = {key: os.getenv(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@dataclass(slots=True)
class TicketFlowRunner:
    settings: TicketFlowSettings
    repository: TicketFlowRepository
    triage_agent: TriageAgent
    knowledge_agent: KnowledgeAgent
    resolution_agent: ResolutionAgent
    tools: TicketTools
    retriever: HybridRetriever
    email_client: MCPEmailClient | None
    attachment_parser: AttachmentParser
    graph: Any
    checkpointer_backend: str = "memory"
    _resource_closers: list[Callable[[], None]] | None = None

    @classmethod
    def from_project_root(cls, project_root: str | Path, overrides: dict[str, str] | None = None) -> "TicketFlowRunner":
        project_root = Path(project_root)
        with _env_overrides(overrides):
            settings = TicketFlowSettings.from_project_root(project_root)

        repository = TicketFlowRepository(settings.db_path)
        seed_dir = settings.seed_dir
        if not (seed_dir / "tickets.csv").exists():
            build_seed_dataset(seed_dir)
        repository.bootstrap(seed_dir)

        cloud_client = None
        if settings.cloud_llm_enabled:
            cloud_client = OpenAICompatClient(
                base_url=settings.cloud_api_base_url,
                api_key=settings.cloud_api_key,
                model=settings.cloud_model_name,
                source_label="llm_cloud",
                chat_path=settings.cloud_api_chat_path,
                verify=settings.cloud_api_ca_cert or True,
                max_tokens=256,
            )

        minimind_client = None
        if settings.minimind_enabled:
            minimind_client = OpenAICompatClient(
                base_url=settings.minimind_base_url,
                api_key=settings.minimind_api_key,
                model=settings.minimind_model,
                source_label="llm_minimind",
                chat_path=settings.minimind_chat_path,
                verify=True,
                max_tokens=192,
                timeout=60,
            )

        minimind_structured_client = None
        if settings.minimind_structured_base_url and settings.minimind_structured_model:
            minimind_structured_client = OpenAICompatClient(
                base_url=settings.minimind_structured_base_url,
                api_key=settings.minimind_structured_api_key,
                model=settings.minimind_structured_model,
                source_label="llm_minimind",
                chat_path=settings.minimind_chat_path,
                verify=True,
                max_tokens=192,
                timeout=60,
            )

        minimind_reply_client = None
        if settings.minimind_reply_base_url and settings.minimind_reply_model:
            minimind_reply_client = OpenAICompatClient(
                base_url=settings.minimind_reply_base_url,
                api_key=settings.minimind_reply_api_key,
                model=settings.minimind_reply_model,
                source_label="llm_minimind",
                chat_path=settings.minimind_chat_path,
                verify=True,
                max_tokens=192,
                timeout=60,
            )

        intent_recognizer = None
        if settings.intent_backend.strip().lower() in {"bert", "auto"}:
            try:
                intent_recognizer = BertIntentRecognizer(
                    model_name=settings.bert_intent_model,
                    device=settings.bert_intent_device,
                )
            except Exception:
                if settings.intent_backend.strip().lower() == "bert":
                    raise

        triage_primary_llm = None
        triage_fallback_llm = None
        action_primary_llm = None
        action_fallback_llm = None
        draft_primary_llm = None
        draft_fallback_llm = None
        if settings.model_backend == "cloud_api":
            triage_primary_llm = cloud_client
            action_primary_llm = cloud_client
            draft_primary_llm = cloud_client
        elif settings.model_backend == "minimind_split":
            triage_primary_llm = minimind_structured_client or cloud_client
            triage_fallback_llm = cloud_client if triage_primary_llm is minimind_structured_client else None
            action_primary_llm = minimind_structured_client or cloud_client
            action_fallback_llm = cloud_client if action_primary_llm is minimind_structured_client else None
            draft_primary_llm = minimind_reply_client or cloud_client
            draft_fallback_llm = cloud_client if draft_primary_llm is minimind_reply_client else None
        elif settings.model_backend == "minimind_api":
            triage_primary_llm = minimind_client if settings.minimind_enable_triage and minimind_client is not None else cloud_client or minimind_client
            triage_fallback_llm = cloud_client if triage_primary_llm is minimind_client else minimind_client
            action_primary_llm = minimind_client or cloud_client
            action_fallback_llm = cloud_client if action_primary_llm is minimind_client else minimind_client
            draft_primary_llm = minimind_client if settings.minimind_enable_draft_reply and minimind_client is not None else cloud_client or minimind_client
            draft_fallback_llm = cloud_client if draft_primary_llm is minimind_client else minimind_client

        tools = TicketTools(repository)
        retriever = HybridRetriever(
            repository=repository,
            rag_db_dir=settings.rag_db_dir,
            top_k=settings.rag_top_k,
            enabled=settings.rag_enabled,
            embed_backend=settings.rag_embed_backend,
            embed_model=settings.rag_embed_model,
            embed_device=settings.rag_embed_device,
            bm25_k1=settings.rag_bm25_k1,
            bm25_b=settings.rag_bm25_b,
            rrf_k=settings.rag_rrf_k,
            reranker_backend=settings.rag_reranker_backend,
            reranker_model=settings.rag_reranker_model,
            reranker_device=settings.rag_reranker_device,
            cache_enabled=settings.rag_cache_enabled,
            redis_url=settings.redis_url,
            cache_ttl_seconds=settings.rag_cache_ttl_seconds,
        )
        retriever.write_manifest()

        email_client = None
        if settings.incident_email_to or settings.kb_ops_email_to:
            env_overrides = {
                "SMTP_HOST": settings.smtp_host or "",
                "SMTP_PORT": str(settings.smtp_port),
                "SMTP_USERNAME": settings.smtp_username or "",
                "SMTP_AUTH_CODE": settings.smtp_auth_code or "",
                "SMTP_USE_TLS": "true" if settings.smtp_use_tls else "false",
                "EMAIL_FROM_NAME": settings.email_from_name,
            }
            email_client = MCPEmailClient(project_root=project_root, env_overrides=env_overrides)

        runner = cls(
            settings=settings,
            repository=repository,
            triage_agent=TriageAgent(
                primary_llm_client=triage_primary_llm,
                fallback_llm_client=triage_fallback_llm,
                intent_recognizer=intent_recognizer,
                sla_risk_review_enabled=settings.sla_risk_review_enabled,
                sla_risk_threshold=settings.sla_risk_threshold,
            ),
            knowledge_agent=KnowledgeAgent(),
            resolution_agent=ResolutionAgent(
                action_primary_llm_client=action_primary_llm,
                action_fallback_llm_client=action_fallback_llm,
                draft_primary_llm_client=draft_primary_llm,
                draft_fallback_llm_client=draft_fallback_llm,
                reply_review_enabled=settings.reply_review_enabled,
                reply_supported_claim_threshold=settings.reply_supported_claim_threshold,
            ),
            tools=tools,
            retriever=retriever,
            email_client=email_client,
            attachment_parser=AttachmentParser(),
            graph=None,
            checkpointer_backend="memory",
            _resource_closers=[],
        )
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("ticketflow.models", "TicketRecord"),
                ("ticketflow.models", "AuditEvent"),
                ("ticketflow.models", "TriageResult"),
                ("ticketflow.models", "RetrievedDoc"),
                ("ticketflow.models", "RetrievalStats"),
                ("ticketflow.models", "AttachmentRecord"),
                ("ticketflow.models", "AttachmentEvidence"),
                ("ticketflow.models", "ContextSufficiencyResult"),
                ("ticketflow.models", "ActionRoutingDecision"),
                ("ticketflow.models", "ActionProposal"),
                ("ticketflow.models", "ToolApprovalPolicy"),
                ("ticketflow.models", "ReviewDecision"),
                ("ticketflow.models", "TicketResponse"),
                ("ticketflow.models", "ReplyFactCheckResult"),
                ("ticketflow.models", "Citation"),
                ("ticketflow.models", "ExternalOpRecord"),
            ]
        )
        checkpointer = build_checkpointer(
            backend=settings.checkpoint_backend,
            sqlite_path=settings.checkpoint_db_path,
            serializer=serializer,
        )
        runner.checkpointer_backend = checkpointer.backend
        if checkpointer.close is not None:
            runner._resource_closers.append(checkpointer.close)
        runner.graph = runner._build_graph(checkpointer.saver)
        return runner

    def list_open_tickets(self, limit: int = 50) -> list[TicketRecord]:
        return self.repository.list_open_tickets(limit=limit)

    def get_ticket(self, ticket_id: str) -> TicketRecord:
        return self.repository.get_ticket(ticket_id)

    def reset_demo_data(self) -> None:
        self.repository.reset_from_seed(self.settings.seed_dir)

    def run_ticket(self, ticket_id: str, thread_id: str | None = None) -> InvocationResult:
        ticket = self.get_ticket(ticket_id)
        thread_id = thread_id or f"thread-{ticket.ticket_id.lower()}-{uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        output = self.graph.invoke({"thread_id": thread_id, "ticket": ticket}, config=config)
        return self._normalize_result(thread_id, output)

    def resume_ticket(self, thread_id: str, decision: ReviewDecision | dict[str, Any]) -> InvocationResult:
        if isinstance(decision, ReviewDecision):
            decision_model = decision
        elif hasattr(decision, "model_dump"):
            decision_model = ReviewDecision.model_validate(decision.model_dump(mode="json"))
        else:
            decision_model = ReviewDecision.model_validate(decision)
        config = {"configurable": {"thread_id": thread_id}}
        output = self.graph.invoke(Command(resume=decision_model.model_dump(mode="json")), config=config)
        return self._normalize_result(thread_id, output)

    def get_snapshot(self, thread_id: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        return {
            "values": snapshot.values,
            "next": list(snapshot.next),
            "tasks": [task.name for task in snapshot.tasks],
        }

    def close(self) -> None:
        for closer in list(self._resource_closers or []):
            try:
                closer()
            except Exception:
                continue
        if self._resource_closers is not None:
            self._resource_closers.clear()

    def _normalize_result(self, thread_id: str, output: dict[str, Any]) -> InvocationResult:
        interrupted = "__interrupt__" in output
        interrupt_payload = None
        if interrupted:
            interrupt_payload = output["__interrupt__"][0].value
        state = dict(output)
        state["thread_id"] = thread_id
        if interrupted:
            state.pop("__interrupt__", None)
        return InvocationResult(state=state, interrupted=interrupted, interrupt_payload=interrupt_payload)

    def _build_graph(self, checkpointer: Any) -> Any:
        builder = StateGraph(TicketFlowState)
        builder.add_node("intake", self._intake_node)
        builder.add_node("parse_attachments", self._parse_attachments_node)
        builder.add_node("triage", self._triage_node)
        builder.add_node("retrieve_context", self._retrieve_context_node)
        builder.add_node("assess_sufficiency", self._assess_sufficiency_node)
        builder.add_node("route_action", self._route_action_node)
        builder.add_node("propose_action", self._propose_action_node)
        builder.add_node("approval_gate", self._approval_gate_node)
        builder.add_node("execute_action", self._execute_action_node)
        builder.add_node("draft_reply", self._draft_reply_node)
        builder.add_node("fact_check_reply", self._fact_check_reply_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_edge(START, "intake")
        builder.add_edge("intake", "parse_attachments")
        builder.add_edge("parse_attachments", "triage")
        builder.add_edge("triage", "retrieve_context")
        builder.add_edge("retrieve_context", "assess_sufficiency")
        builder.add_edge("assess_sufficiency", "route_action")
        builder.add_edge("route_action", "propose_action")
        builder.add_edge("propose_action", "approval_gate")
        builder.add_edge("approval_gate", "execute_action")
        builder.add_edge("execute_action", "draft_reply")
        builder.add_edge("draft_reply", "fact_check_reply")
        builder.add_edge("fact_check_reply", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=checkpointer)

    def _intake_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        detail = f"监督编排器接收工单 {ticket.ticket_id}，并将其送入分诊节点。"
        trace = _append_trace(state, "supervisor", "intake", detail, {"ticket_id": ticket.ticket_id})
        audit_log = _append_audit(state, self.tools, ticket.ticket_id, "supervisor", "intake", detail, {"ticket_id": ticket.ticket_id})
        return {"trace": trace, "audit_log": audit_log, "approval_state": "not_required", "tool_calls": [], "external_ops": []}

    def _parse_attachments_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        attachments = self.repository.list_ticket_attachments(ticket.ticket_id)
        evidence: list[AttachmentEvidence] = []
        for attachment in attachments:
            parsed = self.attachment_parser.parse(attachment)
            evidence.append(parsed)
            self.repository.update_attachment_parse_status(attachment.attachment_id, "parsed")
        self.repository.replace_attachment_evidence(ticket.ticket_id, evidence)
        detail = f"附件解析完成，共处理 {len(attachments)} 个附件，生成 {len(evidence)} 条多模态证据。"
        payload = {
            "attachment_ids": [item.attachment_id for item in attachments],
            "evidence_ids": [item.evidence_id for item in evidence],
            "evidence_types": [item.evidence_type for item in evidence],
        }
        trace = _append_trace(state, "attachment_parser", "parse_attachments", detail, payload)
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "attachment_parser",
            "attachments_parsed",
            detail,
            payload,
        )
        return {"attachments": attachments, "attachment_evidence": evidence, "trace": trace, "audit_log": audit_log}

    def _triage_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = self.triage_agent.analyze(ticket)
        detail = f"分诊完成，类别={triage_result.category}，优先级={triage_result.priority}。"
        trace = _append_trace(state, "triage_agent", "triage", detail, triage_result.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "triage_agent",
            "triage_completed",
            detail,
            triage_result.model_dump(mode="json"),
        )
        return {"triage_result": triage_result, "trace": trace, "audit_log": audit_log}

    def _retrieve_context_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        customer_profile = self.repository.get_customer_profile(ticket.customer_id)
        order = self.repository.get_order_status(ticket.linked_order_id)

        rag_result = self.retriever.retrieve(ticket) if self.settings.rag_enabled else self.retriever.baseline_retrieve(ticket)
        attachment_docs = [attachment_evidence_to_doc(item) for item in state.get("attachment_evidence", [])]
        retrieved_docs = self.knowledge_agent.retrieve(customer_profile, order, [*attachment_docs, *rag_result.docs])
        if attachment_docs:
            rag_result.stats.source_counts["attachment"] = len(attachment_docs)
            for doc in attachment_docs:
                doc_id = f"attachment:{doc.doc_id}"
                if doc_id not in rag_result.stats.retrieved_doc_ids:
                    rag_result.stats.retrieved_doc_ids.append(doc_id)
            rag_result.stats.final_hits += len(attachment_docs)

        tool_calls = list(state.get("tool_calls", []))
        tool_calls.extend(
            [
                {"tool": "get_customer_profile", "input": {"customer_id": ticket.customer_id}},
                {"tool": "get_order_status", "input": {"order_id": ticket.linked_order_id}},
                {"tool": "hybrid_rag_retrieve", "input": {"query": ticket.title, "product": ticket.product}},
            ]
        )
        if attachment_docs:
            tool_calls.append(
                {
                    "tool": "parse_attachments",
                    "input": {"ticket_id": ticket.ticket_id, "attachment_count": len(attachment_docs)},
                }
            )
        policy_hits = self.tools.search_policy_text(
            query=" ".join(
                filter(
                    None,
                    [
                        ticket.title,
                        ticket.body,
                        triage_result.category,
                        ticket.product,
                        customer_profile.customer_tier if customer_profile else "",
                    ],
                )
            ),
            limit=5,
        )
        tool_calls.append({"tool": "search_policy_text", "input": {"query": ticket.title}})
        detail = f"知识 Agent 已为 {triage_result.category} 场景收集 {len(retrieved_docs)} 条上下文证据。"
        trace = _append_trace(
            state,
            "knowledge_agent",
            "retrieve_context",
            detail,
            rag_result.stats.model_dump(mode="json"),
        )
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "knowledge_agent",
            "context_retrieved",
            detail,
            rag_result.stats.model_dump(mode="json"),
        )
        return {
            "retrieved_docs": retrieved_docs,
            "retrieval_stats": rag_result.stats,
            "policy_hits": policy_hits,
            "tool_calls": tool_calls,
            "trace": trace,
            "audit_log": audit_log,
        }

    def _assess_sufficiency_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        customer_profile = self.repository.get_customer_profile(ticket.customer_id)
        order = self.repository.get_order_status(ticket.linked_order_id)
        retrieved_docs = state.get("retrieved_docs", [])
        policy_hits = state.get("policy_hits", [])
        sufficiency_result = self.resolution_agent.assess_sufficiency(
            ticket=ticket,
            triage=triage_result,
            customer=customer_profile,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        detail = f"上下文充分性评估完成，route_family={sufficiency_result.route_family}，sufficient={sufficiency_result.sufficient}。"
        trace = _append_trace(state, "resolution_agent", "assess_sufficiency", detail, sufficiency_result.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "resolution_agent",
            "sufficiency_assessed",
            detail,
            sufficiency_result.model_dump(mode="json"),
        )
        return {"sufficiency_result": sufficiency_result, "trace": trace, "audit_log": audit_log}

    def _route_action_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        sufficiency_result = state["sufficiency_result"]
        action_route = build_action_route(sufficiency_result)
        detail = (
            f"动作路由完成，route_family={action_route.route_family}，"
            f"{'直接进入保守路线' if action_route.forced_action_type else '进入受约束动作生成'}。"
        )
        trace = _append_trace(state, "supervisor", "route_action", detail, action_route.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "supervisor",
            "action_routed",
            detail,
            action_route.model_dump(mode="json"),
        )
        return {"action_route": action_route, "trace": trace, "audit_log": audit_log}

    def _propose_action_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        customer_profile = self.repository.get_customer_profile(ticket.customer_id)
        order = self.repository.get_order_status(ticket.linked_order_id)
        retrieved_docs = state.get("retrieved_docs", [])

        policy_hits = state.get("policy_hits", [])
        sufficiency_result = state["sufficiency_result"]
        action_route = state["action_route"]
        proposed_action = self.resolution_agent.propose_action(
            ticket,
            triage_result,
            customer_profile,
            order,
            policy_hits,
            retrieved_docs,
            sufficiency_result,
            action_route,
        )

        tool_calls = list(state.get("tool_calls", []))
        detail = f"决策 Agent 提出动作 {proposed_action.action_type}，计划调用 {proposed_action.suggested_tool}。"
        trace = _append_trace(state, "resolution_agent", "propose_action", detail, proposed_action.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "resolution_agent",
            "action_proposed",
            detail,
            proposed_action.model_dump(mode="json"),
        )
        return {
            "proposed_action": proposed_action,
            "tool_calls": tool_calls,
            "trace": trace,
            "audit_log": audit_log,
            "approval_state": "pending_review" if proposed_action.requires_approval else "not_required",
        }

    def _approval_gate_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        proposed_action = state["proposed_action"]
        if not proposed_action.requires_approval:
            detail = "当前动作风险较低，跳过人工审批。"
            trace = _append_trace(state, "supervisor", "approval_gate", detail, {"requires_approval": False})
            audit_log = _append_audit(
                state,
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "approval_skipped",
                detail,
                {"requires_approval": False},
            )
            return {"trace": trace, "audit_log": audit_log}

        decision_payload = interrupt(
            {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "proposed_action": proposed_action.model_dump(mode="json"),
                "reason": proposed_action.rationale,
            }
        )
        review_decision = ReviewDecision.model_validate(decision_payload)
        detail = f"人工审批已完成，结果={review_decision.decision}。"
        trace = _append_trace(state, "manager", "approval_gate", detail, review_decision.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "manager",
            "approval_completed",
            detail,
            review_decision.model_dump(mode="json"),
        )
        approval_state = "approved" if review_decision.decision in {"approve", "edit"} else "rejected"
        return {"review_decision": review_decision, "trace": trace, "audit_log": audit_log, "approval_state": approval_state}

    def _execute_action_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        proposed_action = state["proposed_action"]
        triage_result = state["triage_result"]
        review_decision = state.get("review_decision")
        action = review_decision.edited_action if review_decision and review_decision.edited_action else proposed_action

        tool_calls = list(state.get("tool_calls", []))
        if review_decision and review_decision.decision == "reject":
            execution_result = {
                "status": "rejected",
                "tool_name": None,
                "tool_output": {},
                "handoff_required": True,
                "error": "审批人要求转人工继续处理，当前动作不再自动执行。",
            }
            self.repository.update_ticket_status(ticket.ticket_id, "pending_human")
        else:
            execution_result = self._run_action_with_guards(ticket, action, state)
            if execution_result.get("tool_name"):
                tool_calls.append({"tool": execution_result["tool_name"], "input": action.tool_args})

        external_ops = list(state.get("external_ops", []))
        incident_record = self._maybe_send_incident_email(ticket, triage_result, action, execution_result, state)
        trace = _append_trace(state, "supervisor", "execute_action", f"动作执行阶段结束，状态={execution_result['status']}。", execution_result)
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "supervisor",
            "action_executed",
            f"动作执行阶段结束，状态={execution_result['status']}。",
            execution_result,
        )

        if incident_record is not None:
            external_ops.append(incident_record)
            tool_calls.append({"tool": "mcp.send_incident_email", "input": {"ticket_id": ticket.ticket_id}})
            self.tools.save_external_email_delivery(ticket.ticket_id, incident_record)
            trace = _append_trace(
                {"trace": trace},
                "supervisor",
                "execute_action",
                f"外部升级通知已处理，状态={incident_record.status}。",
                incident_record.model_dump(mode="json"),
            )
            audit_log = _append_audit(
                {"audit_log": audit_log},
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "incident_email_dispatched",
                f"升级通知邮件状态={incident_record.status}。",
                incident_record.model_dump(mode="json"),
            )

        return {
            "execution_result": execution_result,
            "tool_calls": tool_calls,
            "trace": trace,
            "audit_log": audit_log,
            "external_ops": external_ops,
        }

    def _run_action_with_guards(self, ticket: TicketRecord, action: ActionProposal, state: TicketFlowState) -> dict[str, Any]:
        if action.action_type == "refund" and not ticket.linked_order_id:
            return {
                "status": "needs_handoff",
                "tool_name": None,
                "tool_output": {},
                "handoff_required": True,
                "error": "缺少订单证据，退款动作被拦截并转人工处理。",
            }
        if action.action_type in {"refund", "close_ticket_without_contact"} and not state.get("policy_hits"):
            return {
                "status": "needs_handoff",
                "tool_name": None,
                "tool_output": {},
                "handoff_required": True,
                "error": "缺少策略依据，敏感动作被拦截并转人工处理。",
            }

        try:
            if action.suggested_tool == "issue_refund_request":
                tool_output = self.tools.issue_refund_request(**action.tool_args)
                self.tools.update_ticket_status(ticket.ticket_id, action.target_status)
            elif action.suggested_tool == "create_escalation":
                tool_output = self.tools.create_escalation(**action.tool_args)
                self.tools.update_ticket_status(ticket.ticket_id, action.target_status)
            elif action.suggested_tool == "update_ticket_status":
                tool_output = self.tools.update_ticket_status(**action.tool_args)
            else:
                tool_output = {}
            return {
                "status": "executed",
                "tool_name": action.suggested_tool,
                "tool_output": tool_output,
                "handoff_required": False,
                "error": None,
            }
        except Exception as exc:
            self.tools.update_ticket_status(ticket.ticket_id, "pending_human")
            return {
                "status": "needs_handoff",
                "tool_name": action.suggested_tool,
                "tool_output": {},
                "handoff_required": True,
                "error": str(exc),
            }

    def _draft_reply_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        proposed_action = state["review_decision"].edited_action if state.get("review_decision") and state["review_decision"].edited_action else state["proposed_action"]
        execution_result = state["execution_result"]
        retrieved_docs = state.get("retrieved_docs", [])
        templates = self.repository.get_reply_templates(triage_result.category, limit=3)
        execution_view = type("ExecutionView", (), execution_result)()
        response = self.resolution_agent.draft_reply(
            ticket=ticket,
            triage=triage_result,
            proposed_action=proposed_action,
            execution_result=execution_view,
            retrieved_docs=retrieved_docs,
            templates=templates,
            review_decision=state.get("review_decision"),
        )
        detail = f"已生成客户回复草稿，建议状态={response.status}。"
        trace = _append_trace(state, "resolution_agent", "draft_reply", detail, response.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "resolution_agent",
            "reply_drafted",
            detail,
            response.model_dump(mode="json"),
        )
        return {"draft_reply": response, "trace": trace, "audit_log": audit_log}

    def _finalize_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        response: TicketResponse = state["draft_reply"]
        self.tools.update_ticket_status(ticket.ticket_id, response.status)

        external_ops = list(state.get("external_ops", []))
        tool_calls = list(state.get("tool_calls", []))
        trace = _append_trace(state, "supervisor", "finalize", f"工单流程完结，最终状态={response.status}。", {"status": response.status})
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "supervisor",
            "ticket_finalized",
            f"工单流程完结，最终状态={response.status}。",
            {"status": response.status},
        )

        kb_record = self._maybe_submit_kb_candidate(ticket, triage_result, response, state)
        if kb_record is not None:
            external_ops.append(kb_record)
            tool_calls.append({"tool": "mcp.submit_kb_candidate_email", "input": {"ticket_id": ticket.ticket_id}})
            self.tools.save_external_email_delivery(ticket.ticket_id, kb_record)
            trace = _append_trace(
                {"trace": trace},
                "supervisor",
                "finalize",
                f"知识候选外部协同已处理，状态={kb_record.status}。",
                kb_record.model_dump(mode="json"),
            )
            audit_log = _append_audit(
                {"audit_log": audit_log},
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "kb_candidate_dispatched",
                f"知识候选邮件状态={kb_record.status}。",
                kb_record.model_dump(mode="json"),
            )

        return {"trace": trace, "audit_log": audit_log, "external_ops": external_ops, "tool_calls": tool_calls}

    # The methods below intentionally override earlier class definitions so the
    # main chain can migrate to the sufficiency-first architecture without being
    # blocked by legacy encoded strings in the original methods.

    def _approval_gate_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        proposed_action = state["proposed_action"]
        sufficiency_result = state.get("sufficiency_result")
        tool_policy = get_tool_approval_policy(proposed_action.suggested_tool)

        if tool_policy.requires_sufficiency and sufficiency_result is not None and not sufficiency_result.sufficient:
            detail = "敏感工具未满足上下文充分性要求，跳过人工审批并等待人工处理。"
            trace = _append_trace(
                state,
                "supervisor",
                "approval_gate",
                detail,
                {
                    "tool_name": proposed_action.suggested_tool,
                    "tool_policy": tool_policy.model_dump(mode="json"),
                    "sufficiency": sufficiency_result.model_dump(mode="json"),
                },
            )
            audit_log = _append_audit(
                state,
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "tool_approval_skipped_for_insufficient_context",
                detail,
                {
                    "tool_policy": tool_policy.model_dump(mode="json"),
                    "sufficiency": sufficiency_result.model_dump(mode="json"),
                },
            )
            return {
                "trace": trace,
                "audit_log": audit_log,
                "approval_state": "not_required",
                "tool_approval_policy": tool_policy,
            }

        if tool_policy.approval_mode == "auto":
            detail = "当前工具命中自动执行策略，跳过人工审批。"
            trace = _append_trace(
                state,
                "supervisor",
                "approval_gate",
                detail,
                {"tool_policy": tool_policy.model_dump(mode="json")},
            )
            audit_log = _append_audit(
                state,
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "tool_approval_skipped",
                detail,
                {"tool_policy": tool_policy.model_dump(mode="json")},
            )
            return {
                "trace": trace,
                "audit_log": audit_log,
                "approval_state": "not_required",
                "tool_approval_policy": tool_policy,
            }

        decision_payload = interrupt(
            {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "proposed_action": proposed_action.model_dump(mode="json"),
                "reason": proposed_action.rationale,
                "tool_policy": tool_policy.model_dump(mode="json"),
                "sufficiency": sufficiency_result.model_dump(mode="json") if sufficiency_result is not None else None,
                "route_family": proposed_action.route_family,
                "required_sources": sufficiency_result.required_sources if sufficiency_result is not None else [],
                "missing_sources": proposed_action.missing_sources,
                "supporting_doc_ids": proposed_action.supporting_doc_ids,
                "tool_name": proposed_action.suggested_tool,
                "tool_args": proposed_action.tool_args,
            }
        )
        review_decision = ReviewDecision.model_validate(decision_payload)
        detail = f"工具级审批已完成，结果={review_decision.decision}。"
        trace = _append_trace(state, "manager", "approval_gate", detail, review_decision.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "manager",
            "tool_approval_completed",
            detail,
            {
                **review_decision.model_dump(mode="json"),
                "tool_policy": tool_policy.model_dump(mode="json"),
            },
        )
        approval_state = "approved" if review_decision.decision in {"approve", "edit"} else "rejected"
        return {
            "review_decision": review_decision,
            "trace": trace,
            "audit_log": audit_log,
            "approval_state": approval_state,
            "tool_approval_policy": tool_policy,
        }

    def _execute_action_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        proposed_action = state["proposed_action"]
        triage_result = state["triage_result"]
        review_decision = state.get("review_decision")
        tool_policy = state.get("tool_approval_policy")
        action = review_decision.edited_action if review_decision and review_decision.edited_action else proposed_action

        tool_calls = list(state.get("tool_calls", []))
        if review_decision and review_decision.decision == "reject":
            execution_result = {
                "status": "rejected",
                "tool_name": None,
                "tool_output": {},
                "handoff_required": True,
                "error": "人工审批拒绝执行敏感工具，当前工单转人工继续处理。",
            }
            self.repository.update_ticket_status(ticket.ticket_id, "pending_human")
        else:
            execution_result = self._run_action_with_guards(ticket, action)
            if execution_result.get("tool_name"):
                tool_calls.append({"tool": execution_result["tool_name"], "input": action.tool_args})

        external_ops = list(state.get("external_ops", []))
        incident_record = self._maybe_send_incident_email(ticket, triage_result, action, execution_result, state)
        trace = _append_trace(
            state,
            "supervisor",
            "execute_action",
            f"动作执行阶段结束，状态={execution_result['status']}。",
            execution_result,
        )
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "supervisor",
            "action_executed",
            f"动作执行阶段结束，状态={execution_result['status']}。",
            {
                **execution_result,
                "tool_policy": tool_policy.model_dump(mode="json") if tool_policy is not None else None,
            },
        )

        if incident_record is not None:
            external_ops.append(incident_record)
            tool_calls.append({"tool": "mcp.send_incident_email", "input": {"ticket_id": ticket.ticket_id}})
            self.tools.save_external_email_delivery(ticket.ticket_id, incident_record)
            trace = _append_trace(
                {"trace": trace},
                "supervisor",
                "execute_action",
                f"外部升级通知已处理，状态={incident_record.status}。",
                incident_record.model_dump(mode="json"),
            )
            audit_log = _append_audit(
                {"audit_log": audit_log},
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "incident_email_dispatched",
                f"升级通知邮件状态={incident_record.status}。",
                incident_record.model_dump(mode="json"),
            )

        return {
            "execution_result": execution_result,
            "tool_calls": tool_calls,
            "trace": trace,
            "audit_log": audit_log,
            "external_ops": external_ops,
            "tool_approval_policy": tool_policy,
        }

    def _run_action_with_guards(self, ticket: TicketRecord, action: ActionProposal) -> dict[str, Any]:
        if action.suggested_tool == "issue_refund_request" and (
            not action.tool_args.get("order_id") or action.tool_args.get("amount") in {None, ""}
        ):
            return {
                "status": "needs_handoff",
                "tool_name": None,
                "tool_output": {},
                "handoff_required": True,
                "error": "Refund tool is missing required execution arguments.",
            }

        try:
            if action.suggested_tool == "issue_refund_request":
                tool_output = self.tools.issue_refund_request(**action.tool_args)
                self.tools.update_ticket_status(ticket.ticket_id, action.target_status)
            elif action.suggested_tool == "create_escalation":
                tool_output = self.tools.create_escalation(**action.tool_args)
                self.tools.update_ticket_status(ticket.ticket_id, action.target_status)
            elif action.suggested_tool == "update_ticket_status":
                tool_output = self.tools.update_ticket_status(**action.tool_args)
            else:
                tool_output = {}
            return {
                "status": "executed",
                "tool_name": action.suggested_tool,
                "tool_output": tool_output,
                "handoff_required": False,
                "error": None,
            }
        except Exception as exc:
            self.tools.update_ticket_status(ticket.ticket_id, "pending_human")
            return {
                "status": "needs_handoff",
                "tool_name": action.suggested_tool,
                "tool_output": {},
                "handoff_required": True,
                "error": str(exc),
            }

    def _draft_reply_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        proposed_action = state["review_decision"].edited_action if state.get("review_decision") and state["review_decision"].edited_action else state["proposed_action"]
        execution_result = state["execution_result"]
        retrieved_docs = state.get("retrieved_docs", [])
        templates = self.repository.get_reply_templates(triage_result.category, limit=3)
        execution_view = type("ExecutionView", (), execution_result)()
        response = self.resolution_agent.draft_reply(
            ticket=ticket,
            triage=triage_result,
            proposed_action=proposed_action,
            execution_result=execution_view,
            retrieved_docs=retrieved_docs,
            templates=templates,
            review_decision=state.get("review_decision"),
        )
        detail = f"已生成客户回复草稿，建议状态={response.status}。"
        trace = _append_trace(state, "resolution_agent", "draft_reply", detail, response.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "resolution_agent",
            "reply_drafted",
            detail,
            response.model_dump(mode="json"),
        )
        return {"reply_draft": response, "trace": trace, "audit_log": audit_log}

    def _fact_check_reply_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        proposed_action = state["review_decision"].edited_action if state.get("review_decision") and state["review_decision"].edited_action else state["proposed_action"]
        execution_result = state["execution_result"]
        retrieved_docs = state.get("retrieved_docs", [])
        reply_draft: TicketResponse = state["reply_draft"]
        execution_view = type("ExecutionView", (), execution_result)()
        checked_reply, fact_check = self.resolution_agent.fact_check_reply(
            ticket=ticket,
            triage=triage_result,
            proposed_action=proposed_action,
            execution_result=execution_view,
            retrieved_docs=retrieved_docs,
            response=reply_draft,
        )
        detail = (
            f"回复事实校验完成，passed={fact_check.passed}，"
            f"rewritten={fact_check.rewritten}，downgraded={fact_check.downgraded}。"
        )
        trace = _append_trace(state, "resolution_agent", "fact_check_reply", detail, fact_check.model_dump(mode="json"))
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "resolution_agent",
            "reply_fact_checked",
            detail,
            fact_check.model_dump(mode="json"),
        )
        if fact_check.rewritten or fact_check.downgraded:
            trace = _append_trace(
                {"trace": trace},
                "resolution_agent",
                "fact_check_reply",
                "回复已根据事实校验结果被重写或降级。",
                checked_reply.model_dump(mode="json"),
            )
            audit_log = _append_audit(
                {"audit_log": audit_log},
                self.tools,
                ticket.ticket_id,
                "resolution_agent",
                "reply_rewritten_or_downgraded",
                "回复已根据事实校验结果被重写或降级。",
                checked_reply.model_dump(mode="json"),
            )
        return {
            "draft_reply": checked_reply,
            "reply_fact_check": fact_check,
            "trace": trace,
            "audit_log": audit_log,
        }

    def _finalize_node(self, state: TicketFlowState) -> dict[str, Any]:
        ticket = state["ticket"]
        triage_result = state["triage_result"]
        response: TicketResponse = state["draft_reply"]
        self.tools.update_ticket_status(ticket.ticket_id, response.status)

        external_ops = list(state.get("external_ops", []))
        tool_calls = list(state.get("tool_calls", []))
        trace = _append_trace(
            state,
            "supervisor",
            "finalize",
            f"工单流程完结，最终状态={response.status}。",
            {"status": response.status},
        )
        audit_log = _append_audit(
            state,
            self.tools,
            ticket.ticket_id,
            "supervisor",
            "ticket_finalized",
            f"工单流程完结，最终状态={response.status}。",
            {
                "status": response.status,
                "reply_fact_check": state.get("reply_fact_check").model_dump(mode="json") if state.get("reply_fact_check") else None,
            },
        )

        kb_record = self._maybe_submit_kb_candidate(ticket, triage_result, response, state)
        if kb_record is not None:
            external_ops.append(kb_record)
            tool_calls.append({"tool": "mcp.submit_kb_candidate_email", "input": {"ticket_id": ticket.ticket_id}})
            self.tools.save_external_email_delivery(ticket.ticket_id, kb_record)
            trace = _append_trace(
                {"trace": trace},
                "supervisor",
                "finalize",
                f"知识候选外部协同已处理，状态={kb_record.status}。",
                kb_record.model_dump(mode="json"),
            )
            audit_log = _append_audit(
                {"audit_log": audit_log},
                self.tools,
                ticket.ticket_id,
                "supervisor",
                "kb_candidate_dispatched",
                f"知识候选邮件状态={kb_record.status}。",
                kb_record.model_dump(mode="json"),
            )

        return {"trace": trace, "audit_log": audit_log, "external_ops": external_ops, "tool_calls": tool_calls}

    def _maybe_send_incident_email(
        self,
        ticket: TicketRecord,
        triage_result: Any,
        action: ActionProposal,
        execution_result: dict[str, Any],
        state: TicketFlowState,
    ) -> ExternalOpRecord | None:
        if action.action_type != "escalation" or execution_result["status"] != "executed":
            return None
        if not self.settings.incident_email_to:
            return ExternalOpRecord(
                op_type="incident_email",
                status="skipped",
                recipient="-",
                subject=f"[TicketFlow 升级通知] {ticket.ticket_id}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error_message="INCIDENT_EMAIL_TO 未配置",
                payload={"reason": "missing_incident_email_to"},
            )
        smtp_issue = self._smtp_configuration_issue() if isinstance(self.email_client, MCPEmailClient) else None
        if smtp_issue:
            return ExternalOpRecord(
                op_type="incident_email",
                status="skipped",
                recipient=self.settings.incident_email_to,
                subject=f"[TicketFlow 升级通知] {ticket.ticket_id}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error_message=smtp_issue,
                payload={"reason": "smtp_not_ready"},
            )
        if self.email_client is None:
            return ExternalOpRecord(
                op_type="incident_email",
                status="failed",
                recipient=self.settings.incident_email_to,
                subject=f"[TicketFlow 升级通知] {ticket.ticket_id}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error_message="MCP email client 未初始化",
                payload={"reason": "missing_email_client"},
            )

        evidence_refs = [citation.source_path for doc in state.get("retrieved_docs", [])[:4] for citation in doc.citations[:1]]
        payload = {
            "ticket_id": ticket.ticket_id,
            "category": triage_result.category,
            "priority": triage_result.priority,
            "sla_risk": triage_result.sla_risk,
            "customer_tier": ticket.customer_tier,
            "action_type": action.action_type,
            "summary": f"{ticket.title} | {ticket.body[:80]}",
            "evidence_refs": evidence_refs or ["无显式证据引用"],
            "recipients": [self.settings.incident_email_to],
        }
        started = time.perf_counter()
        try:
            result = self.email_client.call_tool("send_incident_email", payload)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return ExternalOpRecord(
                op_type="incident_email",
                status="sent",
                recipient=self.settings.incident_email_to,
                subject=f"[TicketFlow 升级通知] {ticket.ticket_id} | {triage_result.category} | {triage_result.priority}",
                provider_message_id=result.get("provider_message_id"),
                delivery_id=result.get("delivery_id"),
                latency_ms=latency_ms,
                created_at=result.get("sent_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                payload=payload,
            )
        except Exception as exc:
            return ExternalOpRecord(
                op_type="incident_email",
                status="failed",
                recipient=self.settings.incident_email_to,
                subject=f"[TicketFlow 升级通知] {ticket.ticket_id} | {triage_result.category} | {triage_result.priority}",
                error_message=str(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                payload=payload,
            )

    def _maybe_submit_kb_candidate(
        self,
        ticket: TicketRecord,
        triage_result: Any,
        response: TicketResponse,
        state: TicketFlowState,
    ) -> ExternalOpRecord | None:
        if not self._should_submit_kb_candidate(ticket, triage_result, state):
            return None
        if not self.settings.kb_ops_email_to:
            return ExternalOpRecord(
                op_type="kb_candidate_email",
                status="skipped",
                recipient="-",
                subject=f"[TicketFlow 知识候选] {ticket.ticket_id}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error_message="KB_OPS_EMAIL_TO 未配置",
                payload={"reason": "missing_kb_ops_email_to"},
            )
        smtp_issue = self._smtp_configuration_issue() if isinstance(self.email_client, MCPEmailClient) else None
        if smtp_issue:
            return ExternalOpRecord(
                op_type="kb_candidate_email",
                status="skipped",
                recipient=self.settings.kb_ops_email_to,
                subject=f"[TicketFlow 知识候选] {ticket.ticket_id}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error_message=smtp_issue,
                payload={"reason": "smtp_not_ready"},
            )
        if self.email_client is None:
            return ExternalOpRecord(
                op_type="kb_candidate_email",
                status="failed",
                recipient=self.settings.kb_ops_email_to,
                subject=f"[TicketFlow 知识候选] {ticket.ticket_id}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                error_message="MCP email client 未初始化",
                payload={"reason": "missing_email_client"},
            )

        retrieved_docs = state.get("retrieved_docs", [])
        evidence_refs = [citation.source_path for doc in retrieved_docs[:4] for citation in doc.citations[:1]]
        gap_reason = self._knowledge_gap_reason(ticket, triage_result, state)
        payload = {
            "ticket_id": ticket.ticket_id,
            "category": triage_result.category,
            "issue_summary": ticket.title,
            "resolution_summary": response.internal_note,
            "knowledge_gap_reason": gap_reason,
            "suggested_kb_title": f"{ticket.title} 处理说明",
            "evidence_refs": evidence_refs or ["无显式证据引用"],
            "recipients": [self.settings.kb_ops_email_to],
        }
        started = time.perf_counter()
        try:
            result = self.email_client.call_tool("submit_kb_candidate_email", payload)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return ExternalOpRecord(
                op_type="kb_candidate_email",
                status="sent",
                recipient=self.settings.kb_ops_email_to,
                subject=f"[TicketFlow 知识候选] {ticket.ticket_id} | {ticket.title}",
                provider_message_id=result.get("provider_message_id"),
                delivery_id=result.get("delivery_id"),
                latency_ms=latency_ms,
                created_at=result.get("sent_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                payload=payload,
            )
        except Exception as exc:
            return ExternalOpRecord(
                op_type="kb_candidate_email",
                status="failed",
                recipient=self.settings.kb_ops_email_to,
                subject=f"[TicketFlow 知识候选] {ticket.ticket_id} | {ticket.title}",
                error_message=str(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                payload=payload,
            )

    def _should_submit_kb_candidate(self, ticket: TicketRecord, triage_result: Any, state: TicketFlowState) -> bool:
        retrieved_docs = state.get("retrieved_docs", [])
        kb_hits = sum(1 for doc in retrieved_docs if doc.source_type == "kb")
        history_hits = sum(1 for doc in retrieved_docs if doc.source_type == "history")
        if kb_hits == 0:
            return True
        if history_hits > 0 and triage_result.priority in {"high", "urgent"}:
            return True
        if ticket.customer_tier == "enterprise" and triage_result.priority in {"high", "urgent"}:
            return True
        return False

    def _smtp_configuration_issue(self) -> str | None:
        missing: list[str] = []
        if not self.settings.smtp_host:
            missing.append("发信服务器")
        if not self.settings.smtp_username:
            missing.append("发件邮箱")
        if not self.settings.smtp_auth_code:
            missing.append("邮箱授权码")
        if not missing:
            return None
        return "外部协同收件人已接入，但真实邮件发送尚未启用；待补充：" + "、".join(missing) + "。"

    def _knowledge_gap_reason(self, ticket: TicketRecord, triage_result: Any, state: TicketFlowState) -> str:
        retrieved_docs = state.get("retrieved_docs", [])
        kb_hits = sum(1 for doc in retrieved_docs if doc.source_type == "kb")
        history_hits = sum(1 for doc in retrieved_docs if doc.source_type == "history")
        if kb_hits == 0:
            return "当前问题没有命中足够知识库证据，需要补充新的 FAQ 或处理指引。"
        if ticket.customer_tier == "enterprise" and triage_result.priority in {"high", "urgent"}:
            return "高价值高风险案例，建议沉淀为重点 SOP 或升级处理模板。"
        if history_hits > 0 and triage_result.priority in {"high", "urgent"}:
            return "同类高优问题存在历史案例，建议沉淀统一处理经验，减少重复人工判断。"
        return "当前问题具备知识沉淀价值。"
