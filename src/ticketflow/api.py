from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .claw import ClawRegistry, ClawRuntime
from .conversation_agent import AgentChatContext, handle_agent_chat
from .graph import TicketFlowRunner
from .knowledge_graph import build_ticket_graph_from_repository, create_knowledge_graph_store
from .models import ActionProposal, ApprovalDecision, ReviewDecision
from .observability import (
    classify_error,
    observe_latency,
    record_metric,
    record_observability_event,
    render_prometheus_metrics,
    set_metric,
    set_trace_id,
)
from .service_settings import ServiceSettings
from .skills import SkillRegistry, SkillRuntime
from .worker import enqueue_deliver_outbox_event, enqueue_pending_outbox_for_ticket, enqueue_run_claw_task, enqueue_run_ticket_workflow


class ApprovalDecisionPayload(BaseModel):
    decision: ApprovalDecision
    reviewer: str = "operator"
    comment: str | None = None
    edited_action: dict[str, Any] | None = None


class WorkflowResumePayload(BaseModel):
    decision: ApprovalDecision
    comment: str | None = None
    edited_action: dict[str, Any] | None = None


class OutboxEventPayload(BaseModel):
    ticket_id: str
    operation_type: str
    business_key: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentChatPayload(BaseModel):
    message: str
    session_id: str | None = None
    actor: str = "operator"


class SkillRunPayload(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    actor: str = "operator"
    ticket_id: str | None = None
    idempotency_key: str | None = None


class ClawRunPayload(BaseModel):
    actor: str = "operator"
    config: dict[str, Any] = Field(default_factory=dict)


def _model_payload(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return jsonable_encoder(model)


def _get_runner(request: Request) -> TicketFlowRunner:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        runner = TicketFlowRunner.from_project_root(
            request.app.state.project_root,
            overrides=request.app.state.runner_overrides,
        )
        request.app.state.runner = runner
    return runner


def _get_kg_store(request: Request):
    kg_store = getattr(request.app.state, "kg_store", None)
    if kg_store is None:
        kg_store = create_knowledge_graph_store(request.app.state.service_settings)
        request.app.state.kg_store = kg_store
    return kg_store


def _get_skill_registry(request: Request) -> SkillRegistry:
    return SkillRegistry(_get_runner(request).repository, request.app.state.service_settings.skills_dir)


def _get_skill_runtime(request: Request) -> SkillRuntime:
    runner = _get_runner(request)
    return SkillRuntime(
        runner.repository,
        project_root=request.app.state.project_root,
        runner_overrides=request.app.state.runner_overrides,
        knowledge_graph_store=_get_kg_store(request),
    )


def _get_claw_registry(request: Request) -> ClawRegistry:
    return ClawRegistry(_get_runner(request).repository, request.app.state.service_settings.claw_tasks_dir)


def _get_claw_runtime(request: Request) -> ClawRuntime:
    return ClawRuntime(
        _get_runner(request).repository,
        project_root=request.app.state.project_root,
        runner_overrides=request.app.state.runner_overrides,
    )


def _ticket_or_404(runner: TicketFlowRunner, ticket_id: str):
    try:
        return runner.get_ticket(ticket_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _summarize_result(ticket_id: str, result) -> dict[str, Any]:
    state = result.state
    ticket = state.get("ticket")
    if hasattr(ticket, "model_dump"):
        ticket_payload = ticket.model_dump(mode="json")
    else:
        ticket_payload = ticket
    return {
        "ticket_id": ticket_id,
        "thread_id": state.get("thread_id"),
        "interrupted": result.interrupted,
        "interrupt_payload": result.interrupt_payload,
        "approval_state": state.get("approval_state"),
        "ticket": ticket_payload,
        "state": jsonable_encoder(state),
    }


def _ops_summary(runner: TicketFlowRunner) -> dict[str, int]:
    tickets = runner.list_open_tickets(limit=1000)
    enterprise_count = sum(1 for ticket in tickets if ticket.customer_tier == "enterprise")
    refund_count = sum(
        1
        for ticket in tickets
        if ticket.expected_category == "billing_refund"
        or "退款" in ticket.title
        or "refund" in ticket.title.lower()
        or "退款" in ticket.body
        or "refund" in ticket.body.lower()
    )
    high_risk_count = sum(
        1
        for ticket in tickets
        if ticket.customer_tier == "enterprise" or ticket.expected_category in {"billing_refund", "technical_issue"}
    )
    return {
        "open_tickets": len(tickets),
        "enterprise_tickets": enterprise_count,
        "risk_watch_tickets": high_risk_count,
        "refund_related_tickets": refund_count,
    }


def _status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _record_component_observation(
    request: Request,
    *,
    span_name: str,
    component: str,
    status: str = "ok",
    latency_ms: int = 0,
    error_type: str | None = None,
    payload_summary: dict[str, object] | None = None,
) -> None:
    settings: ServiceSettings = request.app.state.service_settings
    if not settings.observability_enabled:
        return
    try:
        runner = _get_runner(request)
        record_observability_event(
            runner.repository,
            span_name=span_name,
            component=component,
            status=status,
            latency_ms=latency_ms,
            error_type=error_type,
            payload_summary=payload_summary or {},
        )
    except Exception:
        return


def create_app(
    project_root: str | Path | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> FastAPI:
    root = Path(project_root) if project_root is not None else Path.cwd()
    service_settings = ServiceSettings.from_project_root(root, overrides=runner_overrides)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner = getattr(app.state, "runner", None)
        if runner is not None:
            runner.close()
            app.state.runner = None
        kg_store = getattr(app.state, "kg_store", None)
        if kg_store is not None:
            kg_store.close()
            app.state.kg_store = None

    app = FastAPI(
        title="TicketFlow API",
        version="0.1.0",
        description="Production service skeleton for TicketFlow.",
        lifespan=lifespan,
    )
    app.state.project_root = root
    app.state.service_settings = service_settings
    app.state.runner_overrides = runner_overrides
    app.state.runner = None
    app.state.kg_store = None

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        settings: ServiceSettings = request.app.state.service_settings
        trace_header = settings.trace_header_name or "X-Trace-Id"
        trace_id = set_trace_id(request.headers.get(trace_header))
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            error_type = classify_error(exc, component="api")
            record_metric("ticketflow_errors_total", 1, {"component": "api", "error_type": error_type})
            if settings.observability_enabled:
                try:
                    runner = _get_runner(request)
                    record_observability_event(
                        runner.repository,
                        trace_id=trace_id,
                        span_name=f"{request.method} {request.url.path}",
                        component="api",
                        status="error",
                        latency_ms=latency_ms,
                        error_type=error_type,
                        payload_summary={"path": request.url.path, "method": request.method, "error_message_summary": str(exc)},
                    )
                except Exception:
                    pass
            raise
        finally:
            latency = time.perf_counter() - started
            status_code = getattr(response, "status_code", 500)
            record_metric(
                "ticketflow_api_requests_total",
                1,
                {"method": request.method, "path": request.url.path, "status": str(status_code)},
            )
            observe_latency(
                "ticketflow_api_request_latency_seconds",
                latency,
                {"method": request.method, "path": request.url.path},
            )
            if response is not None:
                response.headers[trace_header] = trace_id
                if settings.observability_enabled:
                    try:
                        runner = _get_runner(request)
                        record_observability_event(
                            runner.repository,
                            trace_id=trace_id,
                            span_name=f"{request.method} {request.url.path}",
                            component="api",
                            status="ok" if status_code < 500 else "error",
                            latency_ms=int(latency * 1000),
                            error_type="api_error" if status_code >= 500 else None,
                            payload_summary={"path": request.url.path, "method": request.method, "status_code": status_code},
                        )
                    except Exception:
                        pass

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "ticketflow-api"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        if not service_settings.prometheus_enabled:
            return "ticketflow_prometheus_enabled 0\n"
        try:
            runner = _get_runner(request)
            set_metric(
                "ticketflow_approvals_pending",
                len(runner.repository.list_approval_requests(status="pending", limit=500)),
            )
            for status, count in _status_counts(runner.repository.list_outbox_events(limit=500)).items():
                set_metric("ticketflow_outbox_events", count, {"status": status})
        except Exception:
            pass
        return render_prometheus_metrics()

    @app.get("/readyz")
    def readyz(request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        return service_settings.readiness_payload(sqlite_db_path=runner.settings.db_path)

    @app.get("/api/v1/observability/summary")
    def observability_summary(request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        events = runner.repository.list_observability_events(limit=1000)
        errors_by_type: dict[str, int] = {}
        events_by_component: dict[str, int] = {}
        for event in events:
            component = str(event.get("component") or "unknown")
            events_by_component[component] = events_by_component.get(component, 0) + 1
            if event.get("status") == "error":
                error_type = str(event.get("error_type") or "unknown_error")
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        workflow_tasks = runner.repository.list_workflow_tasks(limit=500)
        outbox_events = runner.repository.list_outbox_events(limit=500)
        approvals = runner.repository.list_approval_requests(limit=500)
        return {
            "events_count": len(events),
            "events_by_component": events_by_component,
            "errors_by_type": errors_by_type,
            "queues": {
                "workflow_tasks": _status_counts(workflow_tasks),
                "outbox": _status_counts(outbox_events),
                "approvals": _status_counts(approvals),
            },
        }

    @app.get("/api/v1/observability/traces/{trace_id}")
    def observability_trace(trace_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        events = runner.repository.list_observability_events(trace_id=trace_id, limit=500)
        if not events:
            raise HTTPException(status_code=404, detail=f"Unknown trace_id: {trace_id}")
        return {"trace_id": trace_id, "events": events, "count": len(events)}

    @app.get("/api/v1/tickets")
    def list_tickets(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, object]:
        runner = _get_runner(request)
        tickets = [_model_payload(ticket) for ticket in runner.list_open_tickets(limit=limit)]
        return {"tickets": tickets, "count": len(tickets)}

    @app.get("/api/v1/tickets/{ticket_id}")
    def get_ticket(ticket_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        ticket = _ticket_or_404(runner, ticket_id)
        return {"ticket": _model_payload(ticket)}

    @app.post("/api/v1/tickets/{ticket_id}/run", response_model=None)
    def run_ticket(ticket_id: str, request: Request, sync: bool = Query(default=False)) -> Any:
        runner = _get_runner(request)
        ticket = _ticket_or_404(runner, ticket_id)
        try:
            if not sync:
                task = runner.repository.create_workflow_task(
                    ticket_id=ticket.ticket_id,
                    mode="async",
                )
                celery_task_id = enqueue_run_ticket_workflow(
                    task_id=str(task["task_id"]),
                    project_root=request.app.state.project_root,
                    runner_overrides=request.app.state.runner_overrides,
                )
                current_task = runner.repository.get_workflow_task(str(task["task_id"])) or task
                if current_task["status"] in {"queued", "running"}:
                    current_task = runner.repository.set_workflow_task_celery_id(str(task["task_id"]), celery_task_id)
                return JSONResponse(
                    status_code=202,
                    content=jsonable_encoder(
                        {
                            "ticket_id": ticket_id,
                            "task_id": current_task["task_id"],
                            "status": current_task["status"],
                            "celery_task_id": current_task.get("celery_task_id") or celery_task_id,
                            "task": current_task,
                        }
                    ),
                )
            result = runner.run_ticket(ticket_id)
        except Exception as exc:  # pragma: no cover - exercised by integration/runtime paths.
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "workflow_execution_failed",
                    "message": str(exc),
                    "ticket_id": ticket_id,
                },
            ) from exc
        return _summarize_result(ticket_id, result)

    @app.get("/api/v1/tasks")
    def list_tasks(request: Request, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
        runner = _get_runner(request)
        tasks = runner.repository.list_workflow_tasks(limit=limit)
        return {"tasks": tasks, "count": len(tasks)}

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        task = runner.repository.get_workflow_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
        return {"task": task}

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        if runner.repository.get_workflow_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
        task = runner.repository.cancel_workflow_task(task_id, reason="operator_cancelled")
        return {"task": task}

    @app.get("/api/v1/tickets/{ticket_id}/audit")
    def list_ticket_audit(ticket_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        _ticket_or_404(runner, ticket_id)
        events = runner.repository.list_audit_log(ticket_id=ticket_id)
        return {"ticket_id": ticket_id, "events": events, "count": len(events)}

    @app.get("/api/v1/ops/summary")
    def ops_summary(request: Request) -> dict[str, int]:
        runner = _get_runner(request)
        return _ops_summary(runner)

    @app.post("/api/v1/agent/chat")
    def agent_chat(payload: AgentChatPayload, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        _get_skill_registry(request).reload()
        context = AgentChatContext(
            repository=runner.repository,
            project_root=request.app.state.project_root,
            runner_overrides=request.app.state.runner_overrides,
            knowledge_graph_store=_get_kg_store(request),
        )
        try:
            result = handle_agent_chat(payload.message, context)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result["session_id"] = payload.session_id
        result["actor"] = payload.actor
        return result

    @app.get("/api/v1/skills")
    def list_skills(request: Request, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
        runner = _get_runner(request)
        skills = runner.repository.list_agent_skills(limit=limit)
        return {"skills": skills, "count": len(skills)}

    @app.post("/api/v1/skills/reload")
    def reload_skills(request: Request) -> dict[str, object]:
        return _get_skill_registry(request).reload()

    @app.get("/api/v1/skills/runs")
    def list_skill_runs(
        request: Request,
        skill_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        runner = _get_runner(request)
        runs = runner.repository.list_skill_runs(skill_id=skill_id, limit=limit)
        return {"runs": runs, "count": len(runs)}

    @app.get("/api/v1/skills/{skill_id}")
    def get_skill(skill_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        skill = runner.repository.get_agent_skill(skill_id)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Unknown skill_id: {skill_id}")
        return {"skill": skill}

    @app.post("/api/v1/skills/{skill_id}/enable")
    def enable_skill(skill_id: str, request: Request) -> dict[str, object]:
        try:
            skill = _get_runner(request).repository.set_agent_skill_enabled(skill_id, True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"skill": skill}

    @app.post("/api/v1/skills/{skill_id}/disable")
    def disable_skill(skill_id: str, request: Request) -> dict[str, object]:
        try:
            skill = _get_runner(request).repository.set_agent_skill_enabled(skill_id, False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"skill": skill}

    @app.post("/api/v1/skills/{skill_id}/run")
    def run_skill(skill_id: str, payload: SkillRunPayload, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        if runner.repository.get_agent_skill(skill_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown skill_id: {skill_id}")
        result = _get_skill_runtime(request).run_skill(
            skill_id,
            input_payload=payload.input,
            actor=payload.actor,
            ticket_id=payload.ticket_id,
            idempotency_key=payload.idempotency_key,
        )
        _record_component_observation(
            request,
            span_name="skill_run",
            component="skill",
            status="ok" if result.status != "failed" else "error",
            error_type="tool_error" if result.status == "failed" else None,
            payload_summary={"skill_id": skill_id, "run_id": result.run_id, "ticket_id": payload.ticket_id},
        )
        return {"skill_run": result.model_dump(mode="json")}

    @app.get("/api/v1/claw/tasks")
    def list_claw_tasks(request: Request, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
        runner = _get_runner(request)
        tasks = runner.repository.list_claw_tasks(limit=limit)
        return {"tasks": tasks, "count": len(tasks)}

    @app.post("/api/v1/claw/tasks/reload")
    def reload_claw_tasks(request: Request) -> dict[str, object]:
        return _get_claw_registry(request).reload()

    @app.get("/api/v1/claw/tasks/{task_id}")
    def get_claw_task(task_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        task = runner.repository.get_claw_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown Claw task_id: {task_id}")
        return {"task": task}

    @app.post("/api/v1/claw/tasks/{task_id}/run", response_model=None)
    def run_claw_task_endpoint(
        task_id: str,
        request: Request,
        payload: ClawRunPayload | None = None,
        sync: bool = Query(default=False),
        pass_k: int | None = Query(default=None, ge=1, le=10),
    ) -> Any:
        runner = _get_runner(request)
        if runner.repository.get_claw_task(task_id) is None:
            _get_claw_registry(request).reload()
        if runner.repository.get_claw_task(task_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown Claw task_id: {task_id}")
        actor = payload.actor if payload is not None else "operator"
        config = payload.config if payload is not None else {}
        if sync:
            result = _get_claw_runtime(request).run_task(task_id, actor=actor, pass_k=pass_k, config_payload=config)
            average_score = float(result.summary.get("average_score", 0.0))
            _record_component_observation(
                request,
                span_name="claw_run",
                component="claw",
                payload_summary={"task_id": task_id, "run_id": result.run_id, "average_score": average_score},
            )
            record_metric("ticketflow_claw_run_score", average_score, {"task_id": task_id})
            return {"run": result.model_dump(mode="json")}
        task = runner.repository.get_claw_task(task_id)
        selected_pass_k = int(pass_k or (task or {}).get("pass_k") or 3)
        run = runner.repository.create_claw_run(
            task_id=task_id,
            actor=actor,
            pass_k=selected_pass_k,
            config_payload={**config, "sync": False},
            status="queued",
        )
        celery_task_id = enqueue_run_claw_task(
            task_id=task_id,
            project_root=request.app.state.project_root,
            actor=actor,
            pass_k=selected_pass_k,
            run_id=str(run["run_id"]),
            runner_overrides=request.app.state.runner_overrides,
        )
        current_run = runner.repository.get_claw_run(str(run["run_id"])) or run
        if current_run["status"] == "queued":
            current_run = runner.repository.update_claw_run(
                str(run["run_id"]),
                status="queued",
                summary_payload={"celery_task_id": celery_task_id, "pass_k": selected_pass_k},
            )
        _record_component_observation(
            request,
            span_name="claw_run_queued",
            component="claw",
            payload_summary={"task_id": task_id, "run_id": run["run_id"], "celery_task_id": celery_task_id},
        )
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                {
                    "task_id": task_id,
                    "run_id": run["run_id"],
                    "status": current_run["status"],
                    "celery_task_id": celery_task_id,
                    "run": current_run,
                }
            ),
        )

    @app.get("/api/v1/claw/runs/{run_id}")
    def get_claw_run(run_id: str, request: Request) -> dict[str, object]:
        run = _get_runner(request).repository.get_claw_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown Claw run_id: {run_id}")
        return {"run": run}

    @app.get("/api/v1/claw/leaderboard")
    def claw_leaderboard(request: Request, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
        rows = _get_runner(request).repository.list_claw_leaderboard(limit=limit)
        return {"leaderboard": rows, "count": len(rows)}

    @app.get("/api/v1/kg/health")
    def kg_health(request: Request) -> dict[str, object]:
        return _get_kg_store(request).health()

    @app.post("/api/v1/kg/tickets/{ticket_id}/rebuild")
    def rebuild_ticket_graph(
        ticket_id: str,
        request: Request,
        task_id: str | None = Query(default=None),
    ) -> dict[str, object]:
        runner = _get_runner(request)
        _ticket_or_404(runner, ticket_id)
        graph = build_ticket_graph_from_repository(runner.repository, ticket_id, task_id=task_id)
        result = _get_kg_store(request).upsert_ticket_graph(graph)
        _record_component_observation(
            request,
            span_name="kg_rebuild_ticket",
            component="kg",
            payload_summary={
                "ticket_id": ticket_id,
                "task_id": task_id,
                "node_count": graph["node_count"],
                "edge_count": graph["edge_count"],
            },
        )
        return {**result, "graph": {"node_count": graph["node_count"], "edge_count": graph["edge_count"]}}

    @app.get("/api/v1/kg/tickets/{ticket_id}")
    def get_ticket_graph(ticket_id: str, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        _ticket_or_404(runner, ticket_id)
        return _get_kg_store(request).get_ticket_graph(ticket_id)

    @app.get("/api/v1/kg/search")
    def search_kg(
        request: Request,
        q: str = Query(..., min_length=1),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, object]:
        return _get_kg_store(request).search(q, limit=limit)

    @app.get("/api/v1/approvals")
    def list_approvals(
        request: Request,
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        runner = _get_runner(request)
        approvals = runner.repository.list_approval_requests(status=status, limit=limit)
        return {"approvals": approvals, "count": len(approvals)}

    @app.post("/api/v1/approvals/{approval_id}/decision")
    def decide_approval(approval_id: str, payload: ApprovalDecisionPayload, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        existing_approval = runner.repository.get_approval_request(approval_id)
        if existing_approval is None:
            raise HTTPException(status_code=404, detail=f"Unknown approval_id: {approval_id}")
        decision = runner.repository.record_approval_decision(
            approval_id=approval_id,
            decision=payload.decision,
            reviewer=payload.reviewer,
            comment=payload.comment,
            edited_action=payload.edited_action,
        )
        approval = runner.repository.get_approval_request(approval_id)
        _record_component_observation(
            request,
            span_name="approval_decision",
            component="approval",
            payload_summary={"approval_id": approval_id, "decision": payload.decision, "reviewer": payload.reviewer},
        )
        resume_result = None
        approval_payload = existing_approval.get("payload") if existing_approval else {}
        workflow_task_id = approval_payload.get("workflow_task_id") if isinstance(approval_payload, dict) else None
        thread_id = str(existing_approval.get("thread_id")) if existing_approval else ""
        if workflow_task_id and thread_id:
            edited_action = ActionProposal.model_validate(payload.edited_action) if payload.edited_action else None
            review_decision = ReviewDecision(
                decision=payload.decision,
                edited_action=edited_action,
                comment=payload.comment,
            )
            try:
                runner.repository.mark_workflow_task_running(str(workflow_task_id))
                result = runner.resume_ticket(thread_id, review_decision)
                if result.interrupted:
                    next_approval_id = result.interrupt_payload.get("approval_id") if result.interrupt_payload else None
                    task = runner.repository.wait_workflow_task_for_approval(
                        str(workflow_task_id),
                        thread_id=str(result.state["thread_id"]),
                        approval_id=str(next_approval_id) if next_approval_id else None,
                    )
                else:
                    task = runner.repository.complete_workflow_task(
                        str(workflow_task_id),
                        result=result.model_dump(mode="json"),
                    )
                    enqueue_pending_outbox_for_ticket(
                        ticket_id=str(existing_approval["ticket_id"]),
                        project_root=request.app.state.project_root,
                        runner_overrides=request.app.state.runner_overrides,
                    )
                resume_result = {**_summarize_result(str(existing_approval["ticket_id"]), result), "task": task}
            except Exception as exc:
                runner.repository.fail_workflow_task(str(workflow_task_id), error_message=str(exc))
                raise HTTPException(
                    status_code=500,
                    detail={"error": "workflow_resume_failed", "message": str(exc), "thread_id": thread_id},
                ) from exc
        return {"approval": approval, "decision": decision, "resume_result": resume_result}

    @app.post("/api/v1/workflows/{thread_id}/resume")
    def resume_workflow(thread_id: str, payload: WorkflowResumePayload, request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        edited_action = ActionProposal.model_validate(payload.edited_action) if payload.edited_action else None
        review_decision = ReviewDecision(
            decision=payload.decision,
            edited_action=edited_action,
            comment=payload.comment,
        )
        try:
            result = runner.resume_ticket(thread_id, review_decision)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "workflow_resume_failed", "message": str(exc), "thread_id": thread_id},
            ) from exc
        ticket = result.state.get("ticket")
        ticket_id = ticket.ticket_id if hasattr(ticket, "ticket_id") else str(result.state.get("ticket_id", "unknown"))
        return _summarize_result(ticket_id, result)

    @app.post("/api/v1/outbox", response_model=None)
    def create_outbox_event(payload: OutboxEventPayload, request: Request) -> Any:
        runner = _get_runner(request)
        event = runner.repository.create_outbox_event(
            ticket_id=payload.ticket_id,
            operation_type=payload.operation_type,
            business_key=payload.business_key,
            payload=payload.payload,
        )
        celery_task_id = None
        if event["status"] == "pending":
            celery_task_id = enqueue_deliver_outbox_event(
                event_id=str(event["event_id"]),
                project_root=request.app.state.project_root,
                runner_overrides=request.app.state.runner_overrides,
            )
        _record_component_observation(
            request,
            span_name="create_outbox_event",
            component="outbox",
            payload_summary={
                "event_id": event["event_id"],
                "ticket_id": payload.ticket_id,
                "operation_type": payload.operation_type,
                "status": event["status"],
            },
        )
        return JSONResponse(
            status_code=201 if not event["deduplicated"] else 200,
            content=jsonable_encoder({"event": event, "celery_task_id": celery_task_id}),
        )

    @app.get("/api/v1/outbox")
    def list_outbox_events(
        request: Request,
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        runner = _get_runner(request)
        events = runner.repository.list_outbox_events(status=status, limit=limit)
        return {"events": events, "count": len(events)}

    return app


def main() -> None:
    settings = ServiceSettings.from_project_root()
    uvicorn.run(
        create_app(project_root=settings.project_root),
        host=settings.api_host,
        port=settings.api_port,
    )
