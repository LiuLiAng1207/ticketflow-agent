from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .graph import TicketFlowRunner
from .models import ActionProposal, ApprovalDecision, ReviewDecision
from .service_settings import ServiceSettings
from .worker import enqueue_run_ticket_workflow


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


def create_app(
    project_root: str | Path | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> FastAPI:
    root = Path(project_root) if project_root is not None else Path.cwd()
    service_settings = ServiceSettings.from_project_root(root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner = getattr(app.state, "runner", None)
        if runner is not None:
            runner.close()
            app.state.runner = None

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

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "ticketflow-api"}

    @app.get("/readyz")
    def readyz(request: Request) -> dict[str, object]:
        runner = _get_runner(request)
        return service_settings.readiness_payload(sqlite_db_path=runner.settings.db_path)

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
        return JSONResponse(status_code=201 if not event["deduplicated"] else 200, content=jsonable_encoder({"event": event}))

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
