from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents_llm import KnowledgeAgent
from .evals import _binary_f1, _macro_f1, _recall
from .graph import TicketFlowRunner
from .models import TicketRecord


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(text: str | None) -> bool:
    return str(text or "").strip().lower() in {"1", "true", "yes", "y"}


def _target_status_from_action(action_type: str, tool_args: dict[str, Any]) -> str:
    if action_type == "refund":
        return "pending_finance"
    if action_type == "escalation":
        return "escalated"
    if action_type == "request_info":
        return "waiting_on_customer"
    if action_type == "troubleshoot":
        return "investigating"
    return str(tool_args.get("status") or "in_progress")


def _row_to_ticket(row: dict[str, str]) -> TicketRecord:
    created_at = datetime.fromisoformat(row["created_at"])
    return TicketRecord(
        ticket_id=row["ticket_id"],
        channel=row["channel"],
        customer_id=row["customer_id"],
        customer_tier=row["customer_tier"],
        title=row["mapped_title_cn"],
        body=row["mapped_body_cn"],
        product=row["product"],
        created_at=created_at,
        status="open",
        linked_order_id=row.get("linked_order_id") or None,
        source_dataset=row.get("source_dataset") or None,
        source_ticket_ref=row.get("source_ticket_ref") or None,
        source_language=row.get("source_language") or None,
        source_queue=row.get("source_queue") or None,
        source_subject=row.get("source_subject") or None,
        source_body=row.get("source_body") or None,
    )


def _is_annotated(row: dict[str, str]) -> bool:
    required = [
        "gold_category",
        "gold_priority",
        "gold_sla_risk",
        "gold_action_type",
        "gold_approval_required",
        "gold_target_status",
    ]
    status = (row.get("annotation_status") or "").strip().lower()
    if status not in {"annotated", "reviewed", "final"}:
        return False
    return all(str(row.get(field) or "").strip() for field in required)


def evaluate(project_root: Path, gold_path: Path, model_backend: str) -> dict[str, Any]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides={"MODEL_BACKEND": model_backend})
    rows = [row for row in _load_rows(gold_path) if _is_annotated(row)]
    if not rows:
        raise ValueError("Gold 标注文件中没有 status=annotated/reviewed/final 且 gold 字段完整的样本。")

    predicted_categories: list[str] = []
    expected_categories: list[str] = []
    predicted_priorities: list[str] = []
    expected_priorities: list[str] = []
    predicted_sla: list[bool] = []
    expected_sla: list[bool] = []
    predicted_actions: list[str] = []
    expected_actions: list[str] = []
    predicted_approval: list[bool] = []
    expected_approval: list[bool] = []
    predicted_statuses: list[str] = []
    expected_statuses: list[str] = []
    per_case: list[dict[str, Any]] = []

    for row in rows:
        ticket = _row_to_ticket(row)
        triage = runner.triage_agent.analyze(ticket)
        try:
            customer = runner.repository.get_customer_profile(ticket.customer_id)
        except Exception:
            customer = None
        try:
            order = runner.repository.get_order_status(ticket.linked_order_id) if ticket.linked_order_id else None
        except Exception:
            order = None
        rag_result = runner.retriever.retrieve(ticket)
        policy_hits = [doc for doc in rag_result.docs if doc.source_type == "policy"]
        retrieved_docs = KnowledgeAgent().retrieve(customer, order, rag_result.docs)
        action = runner.resolution_agent.propose_action(
            ticket=ticket,
            triage=triage,
            customer=customer,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        predicted_status = str(getattr(action, "target_status", "") or _target_status_from_action(action.action_type, action.tool_args))

        predicted_categories.append(triage.category)
        expected_categories.append(row["gold_category"].strip())
        predicted_priorities.append(triage.priority)
        expected_priorities.append(row["gold_priority"].strip())
        predicted_sla.append(bool(triage.sla_risk))
        expected_sla.append(_truthy(row["gold_sla_risk"]))
        predicted_actions.append(action.action_type)
        expected_actions.append(row["gold_action_type"].strip())
        predicted_approval.append(bool(action.requires_approval))
        expected_approval.append(_truthy(row["gold_approval_required"]))
        predicted_statuses.append(predicted_status)
        expected_statuses.append(row["gold_target_status"].strip())

        per_case.append(
            {
                "sample_id": row["sample_id"],
                "ticket_id": ticket.ticket_id,
                "source_dataset": ticket.source_dataset,
                "gold_category": row["gold_category"],
                "pred_category": triage.category,
                "gold_action_type": row["gold_action_type"],
                "pred_action_type": action.action_type,
                "gold_target_status": row["gold_target_status"],
                "pred_target_status": predicted_status,
                "gold_approval_required": _truthy(row["gold_approval_required"]),
                "pred_approval_required": bool(action.requires_approval),
                "gold_sla_risk": _truthy(row["gold_sla_risk"]),
                "pred_sla_risk": bool(triage.sla_risk),
            }
        )

    action_precision, action_recall, action_f1 = _binary_f1(predicted_approval, expected_approval)
    sla_precision, sla_recall, sla_f1 = _binary_f1(predicted_sla, expected_sla)
    category_accuracy = sum(int(p == e) for p, e in zip(predicted_categories, expected_categories)) / len(rows)
    priority_accuracy = sum(int(p == e) for p, e in zip(predicted_priorities, expected_priorities)) / len(rows)
    action_accuracy = sum(int(p == e) for p, e in zip(predicted_actions, expected_actions)) / len(rows)
    status_accuracy = sum(int(p == e) for p, e in zip(predicted_statuses, expected_statuses)) / len(rows)
    sla_negatives = sum(1 for item in expected_sla if not item)
    sla_false_positives = sum(1 for pred, gold in zip(predicted_sla, expected_sla) if pred and not gold)

    return {
        "sample_size": len(rows),
        "model_backend": model_backend,
        "source_dataset_breakdown": dict(Counter(str(row.get("source_dataset") or "unknown") for row in rows)),
        "metrics": {
            "category_accuracy": round(category_accuracy, 3),
            "category_macro_f1": round(_macro_f1(predicted_categories, expected_categories, sorted(set(expected_categories))), 3),
            "priority_accuracy": round(priority_accuracy, 3),
            "sla_risk_precision": round(sla_precision, 3),
            "sla_risk_recall": round(sla_recall, 3),
            "sla_risk_f1": round(sla_f1, 3),
            "sla_risk_false_positive_rate": round(sla_false_positives / max(sla_negatives, 1), 3),
            "action_type_accuracy": round(action_accuracy, 3),
            "approval_required_recall": round(action_recall, 3),
            "approval_required_precision": round(action_precision, 3),
            "approval_required_f1": round(action_f1, 3),
            "target_status_accuracy": round(status_accuracy, 3),
        },
        "cases": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TicketFlow structured tasks on a manually annotated gold file.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Default: <project_root>/data/gold_eval/public_structured_gold_template.csv",
    )
    parser.add_argument("--model-backend", choices=["rule", "cloud_api", "minimind_api", "minimind_split"], default="rule")
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    gold_path = args.gold_path or (args.project_root / "data" / "gold_eval" / "public_structured_gold_template.csv")
    report = evaluate(args.project_root, gold_path=gold_path, model_backend=args.model_backend)

    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Gold structured evaluation report written to: {args.report_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
