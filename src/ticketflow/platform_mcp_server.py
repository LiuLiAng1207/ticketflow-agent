from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .claw import ClawRegistry, ClawRuntime
from .conversation_agent import AgentChatContext, handle_agent_chat
from .graph import TicketFlowRunner
from .knowledge_graph import build_ticket_graph_from_repository, create_knowledge_graph_store
from .service_settings import ServiceSettings
from .skills import SkillRegistry, SkillRuntime


READ_SCOPE = "read"
EVAL_SCOPE = "eval"
WRITE_SCOPE = "write"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _allowed_scopes() -> set[str]:
    raw = os.getenv("TICKETFLOW_MCP_ALLOWED_SCOPES", "read,eval")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _model_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


@dataclass(slots=True)
class PlatformMCPRuntime:
    project_root: Path
    runner: TicketFlowRunner
    settings: ServiceSettings
    allowed_scopes: set[str]
    write_tools_enabled: bool
    actor: str

    @classmethod
    def from_env(cls) -> "PlatformMCPRuntime":
        project_root = Path(os.getenv("TICKETFLOW_PROJECT_ROOT", Path.cwd())).resolve()
        settings = ServiceSettings.from_project_root(project_root)
        runner = TicketFlowRunner.from_project_root(project_root)
        return cls(
            project_root=project_root,
            runner=runner,
            settings=settings,
            allowed_scopes=_allowed_scopes(),
            write_tools_enabled=_env_bool("TICKETFLOW_MCP_ENABLE_WRITE_TOOLS", False),
            actor=os.getenv("TICKETFLOW_MCP_ACTOR", "mcp-client"),
        )

    def require_scope(self, scope: str) -> None:
        if scope not in self.allowed_scopes:
            raise PermissionError(f"MCP scope 未授权：需要 {scope}，当前允许 {sorted(self.allowed_scopes)}")

    def require_write_enabled(self, tool_name: str) -> None:
        if not self.write_tools_enabled:
            raise PermissionError(
                f"写入类 MCP 工具未启用：{tool_name}。"
                "请设置 TICKETFLOW_MCP_ENABLE_WRITE_TOOLS=true 后再调用。"
            )

    def kg_store(self):
        return create_knowledge_graph_store(self.settings)

    def skill_runtime(self) -> SkillRuntime:
        SkillRegistry(self.runner.repository, self.settings.skills_dir).reload()
        return SkillRuntime(
            repository=self.runner.repository,
            project_root=self.project_root,
            runner_overrides=None,
            knowledge_graph_store=self.kg_store(),
        )

    def claw_runtime(self) -> ClawRuntime:
        ClawRegistry(self.runner.repository, self.settings.claw_tasks_dir).reload()
        SkillRegistry(self.runner.repository, self.settings.skills_dir).reload()
        return ClawRuntime(
            self.runner.repository,
            self.project_root,
            runner_overrides=None,
        )


_RUNTIME: PlatformMCPRuntime | None = None


def _runtime() -> PlatformMCPRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = PlatformMCPRuntime.from_env()
    return _RUNTIME


def create_mcp() -> FastMCP:
    mcp = FastMCP(
        "TicketFlow Platform MCP",
        instructions=(
            "TicketFlow 工单 Agent 平台 MCP 服务。"
            "默认只开放只读和评测能力；写入类能力必须显式启用，并继续走审批、Outbox、Skill 与 Claw 治理链。"
        ),
        host=os.getenv("TICKETFLOW_MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("TICKETFLOW_MCP_PORT", "8000")),
    )

    @mcp.tool(name="ticketflow.ops.summary", description="查询 TicketFlow 运营概览。", structured_output=True)
    def ops_summary() -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        tickets = rt.runner.list_open_tickets(limit=1000)
        return {
            "open_tickets": len(tickets),
            "enterprise_tickets": sum(1 for ticket in tickets if ticket.customer_tier == "enterprise"),
            "risk_watch_tickets": sum(
                1
                for ticket in tickets
                if ticket.customer_tier == "enterprise"
                or ticket.expected_category in {"billing_refund", "technical_issue"}
            ),
            "refund_related_tickets": sum(
                1
                for ticket in tickets
                if ticket.expected_category == "billing_refund" or "refund" in ticket.title.lower()
            ),
        }

    @mcp.tool(name="ticketflow.tickets.list", description="查询开放工单列表。", structured_output=True)
    def tickets_list(limit: int = 50) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        safe_limit = max(1, min(int(limit), 500))
        tickets = [_model_payload(ticket) for ticket in rt.runner.list_open_tickets(limit=safe_limit)]
        return {"tickets": tickets, "count": len(tickets)}

    @mcp.tool(name="ticketflow.tickets.get", description="查询单个工单详情。", structured_output=True)
    def tickets_get(ticket_id: str) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        ticket = rt.runner.get_ticket(ticket_id)
        return {"ticket": _model_payload(ticket)}

    @mcp.tool(name="ticketflow.tickets.audit", description="查询单个工单审计日志。", structured_output=True)
    def tickets_audit(ticket_id: str) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        rt.runner.get_ticket(ticket_id)
        events = rt.runner.repository.list_audit_log(ticket_id=ticket_id)
        return {"ticket_id": ticket_id, "events": events, "count": len(events)}

    @mcp.tool(name="ticketflow.workflow.run", description="提交工单工作流任务。", structured_output=True)
    def workflow_run(ticket_id: str) -> dict[str, Any]:
        rt = _runtime()
        rt.require_write_enabled("ticketflow.workflow.run")
        ticket = rt.runner.get_ticket(ticket_id)
        task = rt.runner.repository.create_workflow_task(ticket_id=ticket.ticket_id, mode="mcp")
        return {
            "ticket_id": ticket.ticket_id,
            "task": task,
            "note": "MCP 只提交受控工作流任务；高风险动作仍需审批，不会绕过治理链。",
        }

    @mcp.tool(name="ticketflow.approvals.list", description="查询审批队列。", structured_output=True)
    def approvals_list(status: str | None = None, limit: int = 100) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        approvals = rt.runner.repository.list_approval_requests(status=status, limit=max(1, min(int(limit), 500)))
        return {"approvals": approvals, "count": len(approvals)}

    @mcp.tool(name="ticketflow.approvals.decide", description="提交人工审批决定。", structured_output=True)
    def approvals_decide(
        approval_id: str,
        decision: str,
        reviewer: str,
        comment: str = "",
        edited_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rt = _runtime()
        rt.require_write_enabled("ticketflow.approvals.decide")
        if decision not in {"approve", "edit", "reject"}:
            raise ValueError("decision 必须是 approve、edit 或 reject")
        existing = rt.runner.repository.get_approval_request(approval_id)
        if existing is None:
            raise KeyError(f"Unknown approval_id: {approval_id}")
        decision_row = rt.runner.repository.record_approval_decision(
            approval_id=approval_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment,
            edited_action=edited_action,
        )
        approval = rt.runner.repository.get_approval_request(approval_id)
        return {
            "approval": approval,
            "decision": decision_row,
            "note": "审批决定已持久化；工作流恢复仍由受控 API/Worker 路径执行。",
        }

    @mcp.tool(name="ticketflow.outbox.list", description="查询 Outbox 投递队列。", structured_output=True)
    def outbox_list(status: str | None = None, limit: int = 100) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        events = rt.runner.repository.list_outbox_events(status=status, limit=max(1, min(int(limit), 500)))
        return {"events": events, "count": len(events)}

    @mcp.tool(name="ticketflow.kg.explain_ticket", description="解释工单证据链和处理路径。", structured_output=True)
    def kg_explain_ticket(ticket_id: str) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        rt.runner.get_ticket(ticket_id)
        graph = build_ticket_graph_from_repository(rt.runner.repository, ticket_id)
        return {
            "ticket_id": ticket_id,
            "graph": graph,
            "note": "知识图谱只用于解释和审计，不作为高风险动作放行依据。",
        }

    @mcp.tool(name="ticketflow.kg.search", description="搜索知识图谱节点。", structured_output=True)
    def kg_search(query: str, limit: int = 20) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        store = rt.kg_store()
        try:
            health = store.health()
            if health.get("status") == "disabled":
                return {
                    "status": "disabled",
                    "backend": health.get("backend", "disabled"),
                    "results": [],
                    "count": 0,
                    "message": "知识图谱后端未启用。",
                }
            result = store.search(query, limit=max(1, min(int(limit), 100)))
            return {"status": "ok", "backend": health.get("backend", "unknown"), **result}
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

    @mcp.tool(name="ticketflow.skills.list", description="查询已注册 Skill。", structured_output=True)
    def skills_list(limit: int = 100) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        SkillRegistry(rt.runner.repository, rt.settings.skills_dir).reload()
        skills = rt.runner.repository.list_agent_skills(limit=max(1, min(int(limit), 500)))
        return {"skills": skills, "count": len(skills)}

    @mcp.tool(name="ticketflow.skills.run", description="运行受控 Skill。", structured_output=True)
    def skills_run(
        skill_id: str,
        input_payload: dict[str, Any] | None = None,
        ticket_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        rt = _runtime()
        rt.require_write_enabled("ticketflow.skills.run")
        result = rt.skill_runtime().run_skill(
            skill_id,
            input_payload=input_payload or {},
            actor=rt.actor,
            ticket_id=ticket_id,
            idempotency_key=idempotency_key,
        )
        return {"skill_run": result.model_dump(mode="json")}

    @mcp.tool(name="ticketflow.claw.tasks.list", description="查询 Claw 评测任务列表。", structured_output=True)
    def claw_tasks_list(limit: int = 100) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(EVAL_SCOPE)
        ClawRegistry(rt.runner.repository, rt.settings.claw_tasks_dir).reload()
        tasks = rt.runner.repository.list_claw_tasks(limit=max(1, min(int(limit), 500)))
        return {"tasks": tasks, "count": len(tasks)}

    @mcp.tool(name="ticketflow.claw.run", description="运行 Claw 评测任务。", structured_output=True)
    def claw_run(task_id: str, pass_k: int = 3) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(EVAL_SCOPE)
        result = rt.claw_runtime().run_task(task_id, actor=rt.actor, pass_k=max(1, min(int(pass_k), 10)))
        return {"run": result.model_dump(mode="json")}

    @mcp.tool(name="ticketflow.claw.runs.get", description="查询 Claw 运行详情。", structured_output=True)
    def claw_runs_get(run_id: str) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(EVAL_SCOPE)
        run = rt.runner.repository.get_claw_run(run_id)
        if run is None:
            raise KeyError(f"Unknown Claw run_id: {run_id}")
        return {"run": run}

    @mcp.tool(name="ticketflow.claw.leaderboard", description="查询 Claw Leaderboard。", structured_output=True)
    def claw_leaderboard(limit: int = 100) -> dict[str, Any]:
        rt = _runtime()
        rt.require_scope(EVAL_SCOPE)
        rows = rt.runner.repository.list_claw_leaderboard(limit=max(1, min(int(limit), 500)))
        return {"leaderboard": rows, "count": len(rows)}

    @mcp.tool(name="ticketflow.agent.chat", description="调用对话式工单 Agent。", structured_output=True)
    def agent_chat(message: str, session_id: str | None = None) -> dict[str, Any]:
        rt = _runtime()
        rt.require_write_enabled("ticketflow.agent.chat")
        context = AgentChatContext(
            repository=rt.runner.repository,
            project_root=rt.project_root,
            runner_overrides=None,
            knowledge_graph_store=rt.kg_store(),
        )
        result = handle_agent_chat(message, context)
        result["session_id"] = session_id
        result["actor"] = rt.actor
        return result

    @mcp.resource(
        "ticketflow://tickets/{ticket_id}",
        name="ticketflow-ticket",
        description="TicketFlow 工单详情 JSON。",
        mime_type="application/json",
    )
    def ticket_resource(ticket_id: str) -> str:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        return _json({"ticket": _model_payload(rt.runner.get_ticket(ticket_id))})

    @mcp.resource(
        "ticketflow://tickets/{ticket_id}/audit",
        name="ticketflow-ticket-audit",
        description="TicketFlow 工单审计日志 JSON。",
        mime_type="application/json",
    )
    def ticket_audit_resource(ticket_id: str) -> str:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        rt.runner.get_ticket(ticket_id)
        events = rt.runner.repository.list_audit_log(ticket_id=ticket_id)
        return _json({"ticket_id": ticket_id, "events": events, "count": len(events)})

    @mcp.resource(
        "ticketflow://kg/tickets/{ticket_id}",
        name="ticketflow-ticket-kg",
        description="TicketFlow 工单知识图谱解释 JSON。",
        mime_type="application/json",
    )
    def ticket_kg_resource(ticket_id: str) -> str:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        rt.runner.get_ticket(ticket_id)
        graph = build_ticket_graph_from_repository(rt.runner.repository, ticket_id)
        return _json({"ticket_id": ticket_id, "graph": graph})

    @mcp.resource(
        "ticketflow://skills/{skill_id}",
        name="ticketflow-skill",
        description="TicketFlow Skill 描述 JSON。",
        mime_type="application/json",
    )
    def skill_resource(skill_id: str) -> str:
        rt = _runtime()
        rt.require_scope(READ_SCOPE)
        SkillRegistry(rt.runner.repository, rt.settings.skills_dir).reload()
        skill = rt.runner.repository.get_agent_skill(skill_id)
        if skill is None:
            raise KeyError(f"Unknown skill_id: {skill_id}")
        return _json({"skill": skill})

    @mcp.resource(
        "ticketflow://claw/runs/{run_id}",
        name="ticketflow-claw-run",
        description="TicketFlow Claw 运行详情 JSON。",
        mime_type="application/json",
    )
    def claw_run_resource(run_id: str) -> str:
        rt = _runtime()
        rt.require_scope(EVAL_SCOPE)
        run = rt.runner.repository.get_claw_run(run_id)
        if run is None:
            raise KeyError(f"Unknown Claw run_id: {run_id}")
        return _json({"run": run})

    @mcp.prompt(name="explain_ticket", description="生成解释某个工单处理原因的提示词。")
    def explain_ticket(ticket_id: str) -> str:
        return (
            f"请基于 TicketFlow 的工单、证据链、审批链和知识图谱解释工单 {ticket_id} 为什么这样处理。"
            "回答时必须区分：证据事实、系统建议、审批状态和不能越权承诺的内容。"
        )

    @mcp.prompt(name="triage_ticket", description="生成工单分诊提示词。")
    def triage_ticket(title: str, body: str, customer_tier: str, product: str) -> str:
        return (
            "请对下面工单进行分诊，输出类别、优先级、SLA 风险和需要检索的证据类型。"
            f"\n标题：{title}\n正文：{body}\n客户等级：{customer_tier}\n产品：{product}"
        )

    @mcp.prompt(name="kb_candidate_from_ticket", description="生成知识候选沉淀提示词。")
    def kb_candidate_from_ticket(ticket_id: str) -> str:
        return (
            f"请根据工单 {ticket_id} 的处理过程提炼知识库候选条目，"
            "包括问题摘要、处理步骤、适用边界、引用证据和不应承诺的内容。"
        )

    @mcp.prompt(name="claw_failure_analysis", description="生成 Claw 失败样例分析提示词。")
    def claw_failure_analysis(run_id: str) -> str:
        return (
            f"请分析 Claw run {run_id} 的失败原因，按 completion、safety、tool correctness、"
            "argument correctness、RAG grounding、approval correctness 和 trajectory quality 分类总结。"
        )

    return mcp


mcp = create_mcp()


def main() -> None:
    transport = os.getenv("TICKETFLOW_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError("TICKETFLOW_MCP_TRANSPORT 必须是 stdio、sse 或 streamable-http")
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
