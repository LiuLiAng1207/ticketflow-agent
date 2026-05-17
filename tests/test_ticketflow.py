from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from types import SimpleNamespace

import ticketflow.agents_llm as agents_llm_module
from ticketflow.agents_llm import (
    ResolutionAgent,
    TriageAgent,
    _build_conservative_local_reply,
    _build_reply_claim_inventory,
    _compute_support_strength,
    _deterministic_reply_fact_check,
    _revalidate_final_reply_against_contract,
    _supporting_history_doc_ids,
)
from ticketflow.evals import run_evaluation
from ticketflow.db import _query_terms
from ticketflow.eval_rag_sensitive import (
    _action_is_acceptable,
    _aggregate_mode_results,
    _available_source_types,
    _expected_sufficient,
    _required_source_types,
    evaluate_rag_sensitive,
)
from ticketflow.eval_retrieval_strategies import _calculate_ranking_metrics
from ticketflow.graph import TicketFlowRunner
from ticketflow.governance import build_action_route, get_tool_approval_policy
from ticketflow.models import ActionProposal, ContextSufficiencyResult, RetrievedDoc, TicketRecord, TicketResponse, TriageResult, ReviewDecision
from ticketflow.rag import BM25CorpusIndex, HybridRetriever, reciprocal_rank_fusion_score


def test_query_terms_extracts_chinese_terms_for_bm25_and_rerank():
    terms = _query_terms("会员无法使用，要求退款，高级会员服务")

    assert "会员无法使用" in terms
    assert "退款" in terms
    assert "高级会员服务" in terms
    assert "会员" in terms
    assert "无法" in terms


class _FakeEmailClient:
    def call_tool(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
        return {
            "status": "sent",
            "delivery_id": f"delivery-{tool_name}",
            "provider_message_id": f"message-{tool_name}",
            "sent_at": "2026-03-31T00:00:00",
            "recipients": payload["recipients"],
        }


class _BrokenEmailClient:
    def call_tool(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError(f"{tool_name} failed")


def _force_escalation_sufficiency_for_email_test(monkeypatch):
    def _assess_sufficiency(self, ticket, triage, customer, order, policy_hits, retrieved_docs):
        del self
        del customer, order
        policy_ids = [f"policy:{doc.doc_id}" for doc in policy_hits[:1]]
        history_ids = [f"history:{doc.doc_id}" for doc in retrieved_docs if doc.source_type == "history"][:1]
        return ContextSufficiencyResult(
            route_family="escalation_candidate",
            risk_level="high" if triage.priority == "urgent" else "elevated",
            required_sources=["policy", "history"],
            supporting_doc_ids=[*policy_ids, *history_ids],
            supporting_history_doc_ids=history_ids,
            missing_sources=[],
            sufficient=True,
            fallback_route=None,
            support_strength=1.0,
            reasoning="email integration test injects sufficient escalation evidence to isolate MCP behavior",
            decision_source="human_edit",
        )

    monkeypatch.setattr(ResolutionAgent, "assess_sufficiency", _assess_sufficiency)


def _find_ticket(runner, category: str, *, missing_order: bool | None = None, tier: str | None = None, refundable: bool | None = None):
    for ticket in runner.list_open_tickets(limit=300):
        if ticket.expected_category != category:
            continue
        if missing_order is True and ticket.linked_order_id:
            continue
        if missing_order is False and not ticket.linked_order_id:
            continue
        if tier is not None and ticket.customer_tier != tier:
            continue
        if refundable is not None:
            order = runner.repository.get_order_status(ticket.linked_order_id)
            if order is None or order.eligible_for_refund != refundable:
                continue
        return ticket
    raise AssertionError(f"Unable to find ticket for category={category}")


def test_refund_ticket_interrupts_before_execution(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False, refundable=True)
    result = runner.run_ticket(ticket.ticket_id, thread_id="refund-test")

    assert result.interrupted is True
    assert result.interrupt_payload["proposed_action"]["action_type"] == "refund"
    assert result.state["approval_state"] == "pending_review"
    assert result.state["triage_result"].category == "billing_refund"
    assert result.state["sufficiency_result"].route_family == "refund_candidate"
    assert result.interrupt_payload["tool_policy"]["tool_name"] == "issue_refund_request"
    assert result.interrupt_payload["sufficiency"]["sufficient"] is True


def test_resume_after_approval_executes_refund(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False, refundable=True)
    initial = runner.run_ticket(ticket.ticket_id, thread_id="refund-approve")
    resumed = runner.resume_ticket(initial.state["thread_id"], ReviewDecision(decision="approve", comment="Proceed"))

    assert resumed.interrupted is False
    assert resumed.state["execution_result"]["status"] == "executed"
    assert resumed.state["draft_reply"].status == "pending_finance"
    assert resumed.state["reply_fact_check"].passed is True


def test_missing_order_refund_requests_more_info(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=True)
    result = runner.run_ticket(ticket.ticket_id, thread_id="refund-missing")

    assert result.interrupted is False
    assert result.state["sufficiency_result"].route_family == "refund_candidate"
    assert result.state["sufficiency_result"].sufficient is False
    assert result.state["proposed_action"].action_type == "request_info"
    assert result.state["proposed_action"].decision_source == "governance"
    assert result.state["draft_reply"].status == "waiting_on_customer"


def test_enterprise_outage_routes_through_escalation_family(runner):
    ticket = _find_ticket(runner, "technical_issue", tier="enterprise")
    result = runner.run_ticket(ticket.ticket_id, thread_id="enterprise-outage")

    assert result.state["sufficiency_result"].route_family == "escalation_candidate"
    assert result.state["triage_result"].priority in {"high", "urgent"}
    if result.interrupted:
        assert result.state["proposed_action"].action_type == "escalation"
        assert result.interrupt_payload["tool_policy"]["tool_name"] == "create_escalation"
    else:
        assert result.state["sufficiency_result"].sufficient is False
        assert result.state["proposed_action"].action_type == "status_update"
        assert result.state["proposed_action"].target_status == "pending_human"


def test_thread_snapshot_preserves_pending_state(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False, refundable=True)
    result = runner.run_ticket(ticket.ticket_id, thread_id="snapshot-thread")

    assert result.interrupted is True
    snapshot = runner.get_snapshot("snapshot-thread")
    assert snapshot["next"] == ["approval_gate"]
    assert snapshot["values"]["ticket"].ticket_id == ticket.ticket_id
    assert snapshot["values"]["sufficiency_result"].route_family == "refund_candidate"


def test_sqlite_checkpoint_persists_across_runner_reloads(ticketflow_project):
    generated_dir = ticketflow_project / "data" / "generated"
    if generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)

    overrides = {
        "RAG_EMBED_BACKEND": "hash",
        "CHECKPOINT_BACKEND": "sqlite",
    }
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides=overrides)
    runner.reset_demo_data()
    ticket = _find_ticket(runner, "billing_refund", missing_order=False, refundable=True)
    result = runner.run_ticket(ticket.ticket_id, thread_id="sqlite-persist-thread")
    assert result.interrupted is True
    runner.close()

    reloaded_runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides=overrides)
    snapshot = reloaded_runner.get_snapshot("sqlite-persist-thread")
    assert snapshot["next"] == ["approval_gate"]
    assert snapshot["values"]["ticket"].ticket_id == ticket.ticket_id

    resumed = reloaded_runner.resume_ticket(
        "sqlite-persist-thread",
        ReviewDecision(decision="approve", comment="resume from sqlite checkpoint"),
    )
    assert resumed.interrupted is False
    assert resumed.state["execution_result"]["status"] == "executed"
    reloaded_runner.close()


def test_rag_retrieve_context_contains_policy_kb_and_history_hits(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False)
    result = runner.run_ticket(ticket.ticket_id, thread_id="rag-coverage")

    stats = result.state["retrieval_stats"]
    assert stats.source_counts.get("kb", 0) >= 1
    assert stats.source_counts.get("policy", 0) >= 1
    assert stats.source_counts.get("history", 0) >= 1


def test_bm25_index_scores_exact_policy_terms_above_unrelated_docs():
    corpus = [
        {
            "id": "policy:refund",
            "document": "退款 订单 支付 审核 财务",
            "metadata": {"source_type": "policy", "doc_id": "POL-R", "title": "退款政策", "snippet": "退款订单审核"},
        },
        {
            "id": "kb:login",
            "document": "登录 密码 验证码 MFA",
            "metadata": {"source_type": "kb", "doc_id": "KB-L", "title": "登录帮助", "snippet": "登录排查"},
        },
    ]
    index = BM25CorpusIndex.from_documents(corpus)

    hits = index.search("订单退款登录", limit=2)

    assert hits[0].doc_id == "POL-R"
    assert hits[0].score > hits[1].score


def test_rrf_gives_overlap_docs_a_stronger_fused_score():
    overlap = reciprocal_rank_fusion_score(bm25_rank=1, vector_rank=2, k=60)
    lexical_only = reciprocal_rank_fusion_score(bm25_rank=1, vector_rank=None, k=60)

    assert overlap > lexical_only


def test_triage_uses_ticket_fields_when_no_llm_or_intent_model(monkeypatch):
    monkeypatch.undo()
    ticket = TicketRecord(
        ticket_id="TCK-OFFLINE-001",
        channel="web",
        customer_id="CUST-001",
        customer_tier="enterprise",
        title="会员扣款后无法使用，要求退款",
        body="我已经付款，但是高级会员服务一直不可用，希望尽快退款。",
        product="高级会员服务",
        created_at=datetime(2026, 5, 15),
        status="open",
        linked_order_id="ORDER-001",
        expected_category="billing_refund",
    )

    result = TriageAgent(primary_llm_client=None, intent_recognizer=None).analyze(ticket)

    assert result.category == "billing_refund"
    assert result.priority == "high"
    assert result.urgency == "same_day"
    assert result.decision_source == "rule"
    assert result.fallback_reason is not None


def test_hybrid_retriever_records_bm25_rrf_and_feature_rerank(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False)
    result = runner.retriever.retrieve(ticket)

    assert result.stats.metadata["lexical_backend"] == "bm25"
    assert result.stats.metadata["fusion_method"] == "rrf"
    assert result.stats.metadata["reranker_backend"] in {"feature", "bge-reranker-v2-m3"}
    assert any("rrf_score" in doc.metadata for doc in result.docs)


def test_hybrid_retriever_keeps_product_kb_recall_lane(runner):
    ticket = runner.get_ticket("TCK-0001")
    query = f"{ticket.title}\n{ticket.body}\n{ticket.product}"
    expected_kb = runner.repository.search_kb(query, ticket.product, limit=1)[0]

    result = runner.retriever.retrieve(ticket)
    retrieved_ids = set(result.stats.retrieved_doc_ids)

    assert f"kb:{expected_kb['doc_id']}" in retrieved_ids


def test_balanced_context_keeps_multiple_product_kb_candidates(runner):
    ticket = runner.get_ticket("TCK-0289")
    query = f"{ticket.title}\n{ticket.body}\n{ticket.product}"
    expected_kb = runner.repository.search_kb(query, ticket.product, limit=1)[0]

    result = runner.retriever.retrieve(ticket)
    retrieved_ids = set(result.stats.retrieved_doc_ids)
    kb_count = sum(1 for doc_id in retrieved_ids if doc_id.startswith("kb:"))

    assert kb_count >= 2
    assert f"kb:{expected_kb['doc_id']}" in retrieved_ids


def test_product_kb_source_rank_survives_semantic_rerank(runner):
    ticket = runner.get_ticket("TCK-0193")
    query = f"{ticket.title}\n{ticket.body}\n{ticket.product}"
    expected_kb = runner.repository.search_kb(query, ticket.product, limit=1)[0]

    result = runner.retriever.retrieve(ticket)
    retrieved_ids = set(result.stats.retrieved_doc_ids)

    assert f"kb:{expected_kb['doc_id']}" in retrieved_ids


def test_balanced_context_selection_preserves_relevance_order(runner):
    docs = [
        RetrievedDoc(doc_id="H-1", source_type="history", title="same case", snippet="same product case", score=0.92, rerank_score=0.92),
        RetrievedDoc(doc_id="H-2", source_type="history", title="near case", snippet="near product case", score=0.82, rerank_score=0.82),
        RetrievedDoc(doc_id="P-1", source_type="policy", title="weak policy", snippet="generic policy", score=0.22, rerank_score=0.22),
        RetrievedDoc(doc_id="K-1", source_type="kb", title="weak kb", snippet="generic kb", score=0.18, rerank_score=0.18),
    ]

    selected = runner.retriever._select_balanced_docs(docs)  # noqa: SLF001

    assert [f"{doc.source_type}:{doc.doc_id}" for doc in selected[:2]] == ["history:H-1", "history:H-2"]
    assert {"policy:P-1", "kb:K-1"}.issubset({f"{doc.source_type}:{doc.doc_id}" for doc in selected})


def test_same_ticket_history_is_prioritized_as_thread_memory():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.top_k = 6
    retriever.bge_reranker = None
    ticket = TicketRecord(
        ticket_id="TCK-036",
        channel="web",
        customer_id="CUST-036",
        customer_tier="enterprise",
        title="Billing refund requires order verification",
        body="The customer reports duplicate billing and needs a refund workflow update.",
        product="Ops Console",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        expected_category="billing_refund",
    )
    same_ticket_history = RetrievedDoc(
        doc_id="HIS-036",
        source_type="history",
        title="Prior handling note for this exact ticket",
        snippet="Thread memory says this ticket is missing a verified order id.",
        score=0.12,
        rerank_score=0.12,
        metadata={"ticket_id": "TCK-036", "product": "Ops Console", "category": "billing_refund"},
    )
    merely_similar_history = RetrievedDoc(
        doc_id="HIS-999",
        source_type="history",
        title="Similar billing refund ticket",
        snippet="Another customer had duplicate billing and requested a refund.",
        score=0.82,
        rerank_score=0.82,
        metadata={"ticket_id": "TCK-999", "product": "Ops Console", "category": "billing_refund"},
    )

    reranked = retriever._rerank_docs(ticket, [merely_similar_history, same_ticket_history])  # noqa: SLF001

    assert reranked[0].doc_id == "HIS-036"
    assert reranked[0].metadata["rerank_exact_ticket_match"] == 1.0


def test_refund_context_keeps_multiple_policy_rules_when_available():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.top_k = 6
    ticket = TicketRecord(
        ticket_id="TCK-REFUND",
        channel="web",
        customer_id="CUST-1",
        customer_tier="enterprise",
        title="Refund request with billing discrepancy",
        body="The customer asks whether refund can proceed and what information is required.",
        product="Ops Console",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        expected_category="billing_refund",
    )
    docs = [
        RetrievedDoc(doc_id="H-1", source_type="history", title="h1", snippet="history", score=0.95, rerank_score=0.95, metadata={}),
        RetrievedDoc(doc_id="H-2", source_type="history", title="h2", snippet="history", score=0.94, rerank_score=0.94, metadata={}),
        RetrievedDoc(doc_id="H-3", source_type="history", title="h3", snippet="history", score=0.93, rerank_score=0.93, metadata={}),
        RetrievedDoc(doc_id="H-4", source_type="history", title="h4", snippet="history", score=0.92, rerank_score=0.92, metadata={}),
        RetrievedDoc(doc_id="K-1", source_type="kb", title="kb", snippet="kb", score=0.91, rerank_score=0.91, metadata={}),
        RetrievedDoc(doc_id="POL-001-01", source_type="policy", title="Refund approval", snippet="Refunds need approval.", score=0.90, rerank_score=0.90, metadata={}),
        RetrievedDoc(doc_id="POL-002-02", source_type="policy", title="Missing order blocks refund", snippet="Do not promise refund without an order id.", score=0.40, rerank_score=0.40, metadata={}),
    ]

    selected = retriever._select_balanced_docs(docs, ticket=ticket)  # noqa: SLF001
    selected_ids = {f"{doc.source_type}:{doc.doc_id}" for doc in selected}

    assert "policy:POL-001-01" in selected_ids
    assert "policy:POL-002-02" in selected_ids


def test_policy_selection_prefers_distinct_policy_rules_over_duplicates():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.top_k = 6
    ticket = TicketRecord(
        ticket_id="TCK-REFUND",
        channel="web",
        customer_id="CUST-1",
        customer_tier="enterprise",
        title="Refund request with verified order",
        body="The customer has a verified order and asks for the refund approval process.",
        product="Ops Console",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        expected_category="billing_refund",
        linked_order_id="ORD-001",
    )
    docs = [
        RetrievedDoc(doc_id="H-1", source_type="history", title="h1", snippet="history", score=0.95, rerank_score=0.95, metadata={}),
        RetrievedDoc(doc_id="H-2", source_type="history", title="h2", snippet="history", score=0.94, rerank_score=0.94, metadata={}),
        RetrievedDoc(doc_id="H-3", source_type="history", title="h3", snippet="history", score=0.93, rerank_score=0.93, metadata={}),
        RetrievedDoc(doc_id="K-1", source_type="kb", title="kb", snippet="kb", score=0.92, rerank_score=0.92, metadata={}),
        RetrievedDoc(doc_id="POL-001-01", source_type="policy", title="Refund approval", snippet="All refunds require approval.", score=0.91, rerank_score=0.91, metadata={}),
        RetrievedDoc(doc_id="POL-001-07", source_type="policy", title="Refund approval", snippet="All refunds require approval.", score=0.90, rerank_score=0.90, metadata={}),
        RetrievedDoc(doc_id="POL-002-02", source_type="policy", title="Missing order blocks refund", snippet="Do not promise refund without a verified order.", score=0.35, rerank_score=0.35, metadata={}),
    ]

    selected = retriever._select_balanced_docs(docs, ticket=ticket)  # noqa: SLF001
    selected_ids = {f"{doc.source_type}:{doc.doc_id}" for doc in selected}

    assert "policy:POL-001-01" in selected_ids
    assert "policy:POL-002-02" in selected_ids
    assert "policy:POL-001-07" not in selected_ids


def test_rag_retrieval_cache_marks_second_read_as_cache_hit(runner_factory):
    runner = runner_factory(RAG_CACHE_ENABLED="true")
    ticket = _find_ticket(runner, "billing_refund", missing_order=False)

    first = runner.retriever.retrieve(ticket)
    second = runner.retriever.retrieve(ticket)

    assert first.stats.metadata["cache_hit"] is False
    assert second.stats.metadata["cache_hit"] is True
    assert second.stats.metadata["cache_backend"] == "memory"


def test_retrieval_metrics_report_hit_rate_mrr_and_error_rate():
    metrics = _calculate_ranking_metrics(
        [
            {"hit": True, "rank": 1},
            {"hit": True, "rank": 4},
            {"hit": False, "rank": None},
        ]
    )

    assert metrics == {"hit_rate": 0.667, "mrr": 0.417, "error_rate": 0.333}


def test_seed_tickets_keep_public_corpus_provenance(runner):
    ticket = runner.list_open_tickets(limit=1)[0]

    assert ticket.source_dataset == "Tobi-Bueck/customer-support-tickets"
    assert ticket.source_ticket_ref
    assert ticket.source_queue
    assert ticket.source_subject


def test_escalation_dispatches_external_emails_when_mcp_is_available(runner_factory, monkeypatch):
    runner = runner_factory(
        INCIDENT_EMAIL_TO="ops@example.com",
        KB_OPS_EMAIL_TO="kb@example.com",
    )
    runner.email_client = _FakeEmailClient()
    _force_escalation_sufficiency_for_email_test(monkeypatch)

    ticket = _find_ticket(runner, "technical_issue", tier="enterprise")
    initial = runner.run_ticket(ticket.ticket_id, thread_id="mcp-success")
    resumed = runner.resume_ticket(initial.state["thread_id"], ReviewDecision(decision="approve", comment="ok"))

    external_ops = resumed.state["external_ops"]
    assert any(op.op_type == "incident_email" and op.status == "sent" for op in external_ops)
    assert any(op.op_type == "kb_candidate_email" and op.status == "sent" for op in external_ops)

    deliveries = runner.repository.list_external_email_deliveries(ticket.ticket_id)
    assert {record.op_type for record in deliveries} >= {"incident_email", "kb_candidate_email"}


def test_external_email_failure_is_non_blocking(runner_factory, monkeypatch):
    runner = runner_factory(
        INCIDENT_EMAIL_TO="ops@example.com",
        KB_OPS_EMAIL_TO="kb@example.com",
    )
    runner.email_client = _BrokenEmailClient()
    _force_escalation_sufficiency_for_email_test(monkeypatch)

    ticket = _find_ticket(runner, "technical_issue", tier="enterprise")
    initial = runner.run_ticket(ticket.ticket_id, thread_id="mcp-failure")
    resumed = runner.resume_ticket(initial.state["thread_id"], ReviewDecision(decision="approve", comment="ok"))

    assert resumed.state["draft_reply"].status == "escalated"
    assert any(op.status == "failed" for op in resumed.state["external_ops"])


def test_evaluation_report_contains_granular_node_metrics(ticketflow_project):
    shutil.rmtree(ticketflow_project / "data" / "generated", ignore_errors=True)
    report = run_evaluation(ticketflow_project, sample_size=8, overrides={"RAG_EMBED_BACKEND": "hash"})

    for key in (
        "category_macro_f1",
        "sla_risk_recall",
        "action_type_accuracy",
        "approval_required_recall",
        "target_status_accuracy",
        "groundedness_rate",
        "policy_violation_rate",
        "hallucination_rate",
        "reply_rubric_proxy_mean",
    ):
        assert key in report


def test_build_action_route_for_refund_insufficiency_forces_request_info():
    route = build_action_route(
        ContextSufficiencyResult(
            route_family="refund_candidate",
            risk_level="high",
            required_sources=["policy", "order"],
            supporting_doc_ids=[],
            missing_sources=["order"],
            sufficient=False,
            fallback_route="request_info",
            reasoning="missing order evidence",
            decision_source="rule",
        )
    )

    assert route.forced_action_type == "request_info"
    assert route.forced_target_status == "waiting_on_customer"
    assert "request_info" in route.allowed_actions


def test_build_action_route_for_escalation_insufficiency_forces_pending_human():
    route = build_action_route(
        ContextSufficiencyResult(
            route_family="escalation_candidate",
            risk_level="high",
            required_sources=["policy", "history"],
            supporting_doc_ids=[],
            missing_sources=["history"],
            sufficient=False,
            fallback_route="status_update",
            reasoning="missing history evidence",
            decision_source="rule",
        )
    )

    assert route.forced_action_type == "status_update"
    assert route.forced_target_status == "pending_human"
    assert "status_update" in route.allowed_actions


def test_build_action_route_for_refund_sufficiency_constrains_to_refund_family():
    route = build_action_route(
        ContextSufficiencyResult(
            route_family="refund_candidate",
            risk_level="high",
            required_sources=["policy", "order"],
            supporting_doc_ids=["policy:POL-002-02", "order:ORD-001"],
            missing_sources=[],
            sufficient=True,
            fallback_route=None,
            reasoning="policy and order evidence are both present",
            decision_source="rule",
        )
    )

    assert route.forced_action_type is None
    assert route.allowed_actions == ["refund"]


def test_build_action_route_for_escalation_sufficiency_constrains_to_escalation_family():
    route = build_action_route(
        ContextSufficiencyResult(
            route_family="escalation_candidate",
            risk_level="high",
            required_sources=["policy", "history"],
            supporting_doc_ids=["policy:POL-005-01", "history:HIS-0207"],
            missing_sources=[],
            sufficient=True,
            fallback_route=None,
            reasoning="policy and history evidence are both present",
            decision_source="rule",
        )
    )

    assert route.forced_action_type is None
    assert route.allowed_actions == ["escalation"]


def test_sensitive_tool_approval_policies_are_explicit():
    refund_policy = get_tool_approval_policy("issue_refund_request")
    escalation_policy = get_tool_approval_policy("create_escalation")
    status_policy = get_tool_approval_policy("update_ticket_status")

    assert refund_policy.sensitive is True
    assert refund_policy.approval_mode == "approve_reject"
    assert escalation_policy.sensitive is True
    assert escalation_policy.approval_mode == "approve_edit_reject"
    assert status_policy.approval_mode == "auto"


def test_rag_sensitive_evaluation_reports_sufficiency_metrics(ticketflow_project):
    eval_csv = Path(__file__).resolve().parents[1] / "data" / "gold_eval" / "rag_sensitive_eval_v1.csv"
    shutil.rmtree(ticketflow_project / "data" / "generated", ignore_errors=True)
    report = evaluate_rag_sensitive(
        project_root=ticketflow_project,
        eval_csv=eval_csv,
        overrides={"RAG_EMBED_BACKEND": "hash"},
    )

    for key in (
        "route_family_accuracy",
        "sufficiency_accuracy",
        "sufficiency_recall_for_high_risk",
        "context_recall",
        "tool_approval_interception_accuracy",
        "reply_fact_check_pass_rate",
        "reply_groundedness_after_rewrite",
    ):
        assert key in report["reports"]["hybrid"]


def test_rag_sensitive_v2_sufficiency_uses_source_requirements_not_exact_doc_ids():
    row = {
        "gold_required_sources": "policy|order",
        "gold_sufficient": "true",
        "gold_policy_doc_id": "POL-002-02",
        "linked_order_id": "ORD-0001",
    }
    docs = [
        RetrievedDoc(doc_id="POL-001-01", source_type="policy", title="policy", snippet="refund policy"),
        RetrievedDoc(doc_id="ORD-0001", source_type="order", title="order", snippet="paid order"),
    ]

    assert _required_source_types(row) == {"policy", "order"}
    assert _available_source_types(docs) == {"policy", "order"}
    assert _expected_sufficient(row, _available_source_types(docs)) is True
    assert _expected_sufficient(row, {"policy"}) is False


def test_rag_sensitive_v2_action_accepts_conservative_fallback_only_when_evidence_missing():
    row = {
        "gold_acceptable_actions": "refund:pending_finance",
        "gold_should_fallback_without_evidence": "true",
        "gold_action_type": "refund",
        "gold_target_status": "pending_finance",
    }
    fallback_action = ActionProposal(
        action_type="request_info",
        target_status="waiting_on_customer",
        rationale="Need more evidence.",
        requires_approval=False,
        suggested_tool="update_ticket_status",
        confidence=1.0,
    )
    refund_action = ActionProposal(
        action_type="refund",
        target_status="pending_finance",
        rationale="Evidence is sufficient.",
        requires_approval=True,
        suggested_tool="issue_refund_request",
        confidence=1.0,
    )

    assert _action_is_acceptable(fallback_action, row, expected_sufficient=False) is True
    assert _action_is_acceptable(fallback_action, row, expected_sufficient=True) is False
    assert _action_is_acceptable(refund_action, row, expected_sufficient=True) is True


def test_evidence_based_action_accuracy_counts_only_sufficiency_passed_evidence_samples():
    base = {
        "seconds": 0.0,
        "route_family_hit": 1,
        "sufficiency_hit": 1,
        "high_risk_sufficiency_candidate": 0,
        "high_risk_sufficiency_hit": 0,
        "policy_required": 0,
        "policy_hit": 0,
        "history_required": 0,
        "history_hit": 0,
        "precision_samples": 0,
        "precision_contribution": 0.0,
        "context_recall_samples": 0,
        "context_recall_contribution": 0.0,
        "reply_usage_sample": 1,
        "reply_usage_hit": 0,
        "reply_claim_coverage": 0.0,
        "reply_draft_fact_check_sample": 1,
        "reply_draft_fact_check_hit": 1,
        "reply_draft_groundedness": 1.0,
        "reply_fact_check_sample": 1,
        "reply_fact_check_hit": 1,
        "reply_groundedness": 1.0,
        "fallback_candidate": 0,
        "fallback_hit": 0,
        "tool_approval_sample": 0,
        "tool_approval_hit": 0,
        "bad_case": None,
    }
    insufficient_fallback = {
        **base,
        "evidence_sample": 0,
        "evidence_action_hit": 0,
        "fallback_candidate": 1,
        "fallback_hit": 1,
    }
    sufficient_correct = {**base, "evidence_sample": 1, "evidence_action_hit": 1}

    report = _aggregate_mode_results(
        mode="hybrid",
        rows=[{}, {}],
        sample_results=[insufficient_fallback, sufficient_correct],
    )

    assert report["evidence_based_action_accuracy"] == 1.0


def test_supporting_history_doc_ids_filters_weak_history_hits():
    strong_history = RetrievedDoc(
        doc_id="HIS-STRONG",
        source_type="history",
        title="Outage playbook",
        snippet="enterprise outage escalation follow-up",
        score=0.61,
        rerank_score=0.74,
        metadata={"rerank_score": 0.74},
    )
    weak_history = RetrievedDoc(
        doc_id="HIS-WEAK",
        source_type="history",
        title="Generic case",
        snippet="status update",
        score=0.22,
        rerank_score=0.33,
        metadata={"rerank_score": 0.33},
    )

    filtered = _supporting_history_doc_ids([weak_history, strong_history], min_history_strength=0.58)

    assert filtered == ["history:HIS-STRONG"]


def test_history_support_strength_uses_calibrated_feature_signals():
    # Real BGE/RRF scores are ranking signals, not calibrated business thresholds.
    # A history case with strong title overlap and product match should pass the
    # escalation evidence gate even when its fused retrieval score is below 0.45.
    strong_history = RetrievedDoc(
        doc_id="HIS-CALIBRATED",
        source_type="history",
        title="订单发货进度长时间未更新",
        snippet="历史经验是先同步订单状态，再判断是否需要升级物流处理。",
        score=0.37,
        rerank_score=0.37,
        metadata={
            "rerank_score": 0.37,
            "rerank_text_overlap": 0.40,
            "rerank_title_overlap": 1.0,
            "rerank_metadata_overlap": 0.40,
            "rerank_exact_product_match": 1.0,
        },
    )

    filtered = _supporting_history_doc_ids([strong_history], min_history_strength=0.45)
    support_strength = _compute_support_strength(
        required_sources=["policy", "history"],
        order=None,
        policy_hits=[
            RetrievedDoc(
                doc_id="POL-1",
                source_type="policy",
                title="物流长期延迟应升级",
                snippet="发货或运输状态停滞超过五天时，应创建物流升级单。",
                score=0.91,
            )
        ],
        retrieved_docs=[strong_history],
        min_history_strength=0.45,
    )

    assert filtered == ["history:HIS-CALIBRATED"]
    assert support_strength >= 0.75


def test_compute_support_strength_uses_strong_history_threshold():
    strong_history = RetrievedDoc(
        doc_id="HIS-STRONG",
        source_type="history",
        title="Outage playbook",
        snippet="enterprise outage escalation follow-up",
        score=0.61,
        rerank_score=0.74,
        metadata={"rerank_score": 0.74},
    )
    support_strength = _compute_support_strength(
        required_sources=["policy", "history"],
        order=None,
        policy_hits=[
            RetrievedDoc(
                doc_id="POL-1",
                source_type="policy",
                title="Escalation policy",
                snippet="sev1 enterprise incident must escalate",
                score=0.91,
            )
        ],
        retrieved_docs=[strong_history],
        min_history_strength=0.58,
    )

    assert support_strength == 0.87


def test_llm_sufficiency_cannot_mark_present_required_source_missing(monkeypatch):
    def _fake_call_structured_json(*args, **kwargs):
        return {
            "sufficient": False,
            "supporting_doc_ids": [],
            "missing_sources": ["policy"],
            "reasoning": "LLM incorrectly marked an available policy source as missing.",
        }

    monkeypatch.setattr(agents_llm_module, "_call_structured_json", _fake_call_structured_json)
    agent = ResolutionAgent(action_primary_llm_client=SimpleNamespace(source_label="llm_cloud"))
    ticket = TicketRecord(
        ticket_id="TCK-ESC",
        channel="web",
        customer_id="CUS-001",
        customer_tier="enterprise",
        title="订单发货进度长时间未更新",
        body="企业客户反馈订单长时间没有物流进展，需要升级处理。",
        product="云桌面协作",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        linked_order_id="ORD-001",
        expected_category="delivery_issue",
    )
    triage = TriageResult(
        category="delivery_issue",
        priority="high",
        urgency="same_day",
        sla_risk=False,
        confidence=1.0,
        reasoning="gold",
    )
    policy = RetrievedDoc(
        doc_id="POL-ESC",
        source_type="policy",
        title="物流长期延迟应升级",
        snippet="发货或运输状态停滞超过五天时，应创建物流升级单。",
        score=1.0,
        metadata={"action_type": "escalation"},
    )
    history = RetrievedDoc(
        doc_id="HIS-ESC",
        source_type="history",
        title="订单发货进度长时间未更新",
        snippet="历史经验是先同步订单状态，再判断是否需要升级物流处理。",
        score=0.37,
        rerank_score=0.37,
        metadata={
            "rerank_score": 0.37,
            "rerank_text_overlap": 0.40,
            "rerank_title_overlap": 1.0,
            "rerank_metadata_overlap": 0.40,
            "rerank_exact_product_match": 1.0,
        },
    )

    result = agent.assess_sufficiency(
        ticket=ticket,
        triage=triage,
        customer=None,
        order=None,
        policy_hits=[policy],
        retrieved_docs=[policy, history],
    )

    assert result.sufficient is True
    assert result.missing_sources == []
    assert result.supporting_doc_ids == ["policy:POL-ESC", "history:HIS-ESC"]


def test_sensitive_tool_approval_policy_overrides_llm_requires_approval_false():
    agent = ResolutionAgent()
    ticket = TicketRecord(
        ticket_id="TCK-ESC",
        channel="web",
        customer_id="CUS-001",
        customer_tier="enterprise",
        title="企业故障需要升级",
        body="生产环境受影响，请尽快升级专家团队。",
        product="运维控制台",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        expected_category="technical_issue",
    )
    triage = TriageResult(
        category="technical_issue",
        priority="urgent",
        urgency="sev1",
        sla_risk=True,
        confidence=1.0,
        reasoning="gold",
    )

    action = agent._build_action_proposal(
        action_type="escalation",
        ticket=ticket,
        triage=triage,
        order=None,
        rationale="LLM proposed escalation but incorrectly disabled approval.",
        confidence=0.8,
        target_status="escalated",
        requires_approval=False,
        route_family="escalation_candidate",
        sufficiency_passed=True,
        supporting_doc_ids=["policy:POL-003-03", "history:HIS-001"],
        missing_sources=[],
    )

    assert action.suggested_tool == "create_escalation"
    assert action.requires_approval is True


def test_sufficient_standard_resolution_does_not_accept_unnecessary_request_info(monkeypatch):
    def fake_structured_json(*args, **kwargs):
        del args, kwargs
        return {
            "action_type": "request_info",
            "target_status": "waiting_on_customer",
            "requires_approval": False,
            "rationale": "The model asked for more information even though the standard route was sufficient.",
            "confidence": 0.6,
        }

    monkeypatch.setattr(agents_llm_module, "_call_structured_json", fake_structured_json)
    agent = ResolutionAgent(action_primary_llm_client=SimpleNamespace(source_label="llm_cloud"))
    ticket = TicketRecord(
        ticket_id="TCK-STANDARD",
        channel="web",
        customer_id="CUST-1",
        customer_tier="standard",
        title="报价与套餐方案咨询，希望尽快回复",
        body="我们想了解产品能力边界、上线节奏和最佳实践。",
        product="视觉巡检仪",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        expected_category="general_inquiry",
    )
    triage = TriageResult(
        category="general_inquiry",
        priority="medium",
        urgency="standard",
        sla_risk=False,
        confidence=0.9,
        reasoning="standard route has enough context",
    )
    sufficiency = ContextSufficiencyResult(
        route_family="standard_resolution",
        risk_level="standard",
        required_sources=[],
        supporting_doc_ids=["history:HIS-1"],
        supporting_history_doc_ids=["history:HIS-1"],
        missing_sources=[],
        sufficient=True,
        fallback_route=None,
        support_strength=1.0,
        reasoning="Standard route has enough context.",
        decision_source="governance",
    )
    route = build_action_route(sufficiency)

    proposal = agent._propose_action_with_llm(  # noqa: SLF001
        agent.action_primary_llm_client,
        ticket,
        triage,
        None,
        None,
        [],
        [],
        sufficiency,
        route,
    )

    assert proposal.action_type == "status_update"
    assert proposal.target_status == "in_progress"
    assert proposal.decision_source == "governance"


def _minimal_ticket() -> TicketRecord:
    return TicketRecord(
        ticket_id="TCK-TEST",
        channel="email",
        customer_id="CUS-001",
        customer_tier="enterprise",
        title="申请退款",
        body="订单存在账单异常，希望申请退款。",
        product="安全金库",
        created_at=datetime(2026, 4, 1, 10, 0, 0),
        status="open",
        linked_order_id="ORD-001",
    )


def _minimal_triage() -> TriageResult:
    return TriageResult(
        category="billing_refund",
        priority="high",
        urgency="same_day",
        sla_risk=False,
        confidence=1.0,
        reasoning="test",
        decision_source="human_edit",
    )


def test_deterministic_reply_fact_check_accepts_supported_conservative_refund_reply():
    ticket = _minimal_ticket()
    triage = _minimal_triage()
    action = ActionProposal(
        action_type="refund",
        target_status="pending_finance",
        rationale="refund route has sufficient evidence",
        requires_approval=True,
        suggested_tool="issue_refund_request",
        tool_args={"ticket_id": "TCK-TEST", "order_id": "ORD-001", "amount": 100.0},
        confidence=0.9,
        route_family="refund_candidate",
        sufficiency_passed=True,
        supporting_doc_ids=["policy:POL-002-02", "order:ORD-001"],
        missing_sources=[],
        decision_source="governance",
    )
    reply = TicketResponse(
        customer_reply="我们已经收到您的退款相关诉求，当前工单已进入退款审核流程，后续进展会尽快同步给您。",
        internal_note="test",
        status="pending_finance",
    )
    execution_view = type("ExecutionView", (), {"status": "executed", "tool_output": {}})()
    claim_inventory = _build_reply_claim_inventory(
        ticket,
        triage,
        action,
        execution_view,
        [],
    )
    fact_check = _deterministic_reply_fact_check(
        reply.customer_reply,
        claim_inventory,
        threshold=0.7,
    )

    assert fact_check["decisive"] is True
    assert fact_check["review_passed"] is True
    assert fact_check["grounded"] is True
    assert float(fact_check["supported_claim_ratio"]) >= 0.7


def test_deterministic_reply_fact_check_rejects_forbidden_refund_completion_claim():
    ticket = _minimal_ticket()
    triage = _minimal_triage()
    action = ActionProposal(
        action_type="refund",
        target_status="pending_finance",
        rationale="refund route has sufficient evidence",
        requires_approval=True,
        suggested_tool="issue_refund_request",
        tool_args={"ticket_id": "TCK-TEST", "order_id": "ORD-001", "amount": 100.0},
        confidence=0.9,
        route_family="refund_candidate",
        sufficiency_passed=True,
        supporting_doc_ids=["policy:POL-002-02", "order:ORD-001"],
        missing_sources=[],
        decision_source="governance",
    )
    reply = TicketResponse(
        customer_reply="您的订单已经退款成功，款项会立即退回原支付账户。",
        internal_note="test",
        status="pending_finance",
    )
    execution_view = type("ExecutionView", (), {"status": "executed", "tool_output": {}})()
    claim_inventory = _build_reply_claim_inventory(
        ticket,
        triage,
        action,
        execution_view,
        [],
    )
    fact_check = _deterministic_reply_fact_check(
        reply.customer_reply,
        claim_inventory,
        threshold=0.7,
    )

    assert fact_check["decisive"] is True
    assert fact_check["review_passed"] is False
    assert fact_check["policy_safe"] is False
    assert fact_check["hallucination_risk"] is True


def test_reply_claim_inventory_binds_refund_contract_to_pending_finance_status():
    ticket = _minimal_ticket()
    triage = _minimal_triage()
    action = ActionProposal(
        action_type="refund",
        target_status="pending_finance",
        rationale="refund route has sufficient evidence",
        requires_approval=True,
        suggested_tool="issue_refund_request",
        tool_args={"ticket_id": "TCK-TEST", "order_id": "ORD-001", "amount": 100.0},
        confidence=0.9,
        route_family="refund_candidate",
        sufficiency_passed=True,
        supporting_doc_ids=["policy:POL-002-02", "order:ORD-001"],
        missing_sources=[],
        decision_source="governance",
    )
    execution_view = type("ExecutionView", (), {"status": "executed", "tool_output": {}})()
    claim_inventory = _build_reply_claim_inventory(
        ticket,
        triage,
        action,
        execution_view,
        [],
    )

    assert "退款审核流程" in claim_inventory["required_claims"]
    assert any("退款审核" in item for item in claim_inventory["supported_claims"])
    assert any("待财务处理" in item or "退款成功" in item for item in claim_inventory["status_contract"])
    assert "退款审核流程" in claim_inventory["template_reply"]


def test_reply_claim_inventory_requests_specific_order_info_when_refund_evidence_missing():
    ticket = _minimal_ticket()
    triage = _minimal_triage()
    action = ActionProposal(
        action_type="request_info",
        target_status="waiting_on_customer",
        rationale="refund evidence missing",
        requires_approval=False,
        suggested_tool="update_ticket_status",
        tool_args={},
        confidence=0.7,
        route_family="refund_candidate",
        sufficiency_passed=False,
        supporting_doc_ids=[],
        missing_sources=["order", "policy"],
        decision_source="governance",
    )
    execution_view = type("ExecutionView", (), {"status": None, "tool_output": {}})()
    claim_inventory = _build_reply_claim_inventory(
        ticket,
        triage,
        action,
        execution_view,
        [],
    )

    assert "请您补充订单号" in claim_inventory["supported_claims"]
    assert "请您补充" in claim_inventory["required_claims"]
    assert "订单号" in claim_inventory["template_reply"]


def test_revalidate_final_reply_accepts_conservative_refund_template():
    ticket = _minimal_ticket()
    triage = _minimal_triage()
    action = ActionProposal(
        action_type="refund",
        target_status="pending_finance",
        rationale="refund route has sufficient evidence",
        requires_approval=True,
        suggested_tool="issue_refund_request",
        tool_args={"ticket_id": "TCK-TEST", "order_id": "ORD-001", "amount": 100.0},
        confidence=0.9,
        route_family="refund_candidate",
        sufficiency_passed=True,
        supporting_doc_ids=["policy:POL-002-02", "order:ORD-001"],
        missing_sources=[],
        decision_source="governance",
    )
    execution_view = type("ExecutionView", (), {"status": "executed", "tool_output": {}})()
    response = TicketResponse(
        customer_reply=_build_conservative_local_reply(ticket, triage, action, execution_view),
        internal_note="test",
        status="pending_finance",
    )

    fact_check = _revalidate_final_reply_against_contract(
        ticket,
        triage,
        action,
        execution_view,
        [],
        response,
        threshold=0.7,
    )

    assert fact_check["decisive"] is True
    assert fact_check["review_passed"] is True
    assert float(fact_check["supported_claim_ratio"]) == 1.0
