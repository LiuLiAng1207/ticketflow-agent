from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import TicketFlowRunner
from .models import ReviewDecision, TicketRecord


def _pick_ticket(
    runner: TicketFlowRunner,
    category: str,
    *,
    missing_order: bool | None = None,
    tier: str | None = None,
    refundable: bool | None = None,
) -> TicketRecord:
    for ticket in runner.list_open_tickets(limit=500):
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
    raise LookupError(f"Unable to find a matching ticket for category={category}")


def run_demo(
    project_root: Path,
    reset_data: bool = False,
    overrides: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides=overrides)
    if reset_data:
        runner.reset_demo_data()

    scenarios = [
        ("refund_approval", _pick_ticket(runner, "billing_refund", missing_order=False, refundable=True)),
        ("missing_order_guardrail", _pick_ticket(runner, "billing_refund", missing_order=True)),
        ("enterprise_escalation", _pick_ticket(runner, "technical_issue", tier="enterprise")),
        ("delivery_monitoring", _pick_ticket(runner, "delivery_issue", missing_order=False)),
    ]

    outputs: list[dict[str, object]] = []
    for name, ticket in scenarios:
        result = runner.run_ticket(ticket.ticket_id, thread_id=f"demo-{name}")
        state = result.state
        if result.interrupted:
            resumed = runner.resume_ticket(
                state["thread_id"],
                ReviewDecision(decision="approve", comment="Auto-approved for CLI demo."),
            )
            state = resumed.state
        outputs.append(
            {
                "scenario": name,
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "customer_tier": ticket.customer_tier,
                "expected_category": ticket.expected_category,
                "predicted_category": state["triage_result"].category,
                "triage_source": state["triage_result"].decision_source,
                "proposed_action": state["proposed_action"].action_type,
                "action_source": state["proposed_action"].decision_source,
                "approval_state": state.get("approval_state"),
                "final_status": state["draft_reply"].status,
                "reply_source": state["draft_reply"].draft_source,
                "trace_steps": len(state.get("trace", [])),
                "rag_source_counts": getattr(state.get("retrieval_stats"), "source_counts", {}),
                "external_ops": [
                    {
                        "op_type": op.op_type,
                        "status": op.status,
                        "recipient": op.recipient,
                    }
                    for op in state.get("external_ops", [])
                ],
                "customer_reply_preview": state["draft_reply"].customer_reply[:160],
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canned TicketFlow demo scenarios.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TicketFlow project root.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Reload the seeded SQLite database before running the demo scenarios.",
    )
    parser.add_argument(
        "--model-backend",
        choices=["rule", "cloud_api", "minimind_api", "minimind_split"],
        default="rule",
        help="演示时使用的模型后端。",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_demo(
                args.project_root,
                reset_data=args.reset_data,
                overrides={"MODEL_BACKEND": args.model_backend},
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
