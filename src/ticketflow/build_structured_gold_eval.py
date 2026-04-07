from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents import KnowledgeAgent
from .graph import TicketFlowRunner
from .models import TicketRecord


OUTPUT_FIELDS = [
    "sample_id",
    "annotation_status",
    "annotator",
    "review_round",
    "notes",
    "ticket_id",
    "customer_id",
    "channel",
    "customer_tier",
    "product",
    "created_at",
    "linked_order_id",
    "source_dataset",
    "source_ticket_ref",
    "source_language",
    "source_queue",
    "source_subject",
    "source_body",
    "mapped_title_cn",
    "mapped_body_cn",
    "silver_category",
    "silver_priority",
    "silver_sla_risk",
    "silver_action_type",
    "silver_approval_required",
    "silver_target_status",
    "gold_category",
    "gold_priority",
    "gold_sla_risk",
    "gold_action_type",
    "gold_approval_required",
    "gold_target_status",
]


def _load_seed_tickets(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _balanced_sample(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("expected_category") or "general_inquiry").strip()].append(row)

    categories = sorted(grouped)
    if not categories:
        return []

    import random

    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    per_category = max(sample_size // len(categories), 1)
    selected: list[dict[str, str]] = []

    for category in categories:
        selected.extend(grouped[category][:per_category])

    leftovers: list[dict[str, str]] = []
    for category in categories:
        leftovers.extend(grouped[category][per_category:])
    rng.shuffle(leftovers)

    remaining = max(sample_size - len(selected), 0)
    selected.extend(leftovers[:remaining])
    return selected[:sample_size]


def _row_to_ticket(row: dict[str, str]) -> TicketRecord:
    created_at = datetime.fromisoformat(row["created_at"])
    return TicketRecord(
        ticket_id=row["ticket_id"],
        channel=row["channel"],
        customer_id=row["customer_id"],
        customer_tier=row["customer_tier"],
        title=row["title"],
        body=row["body"],
        product=row["product"],
        created_at=created_at,
        status=row["status"],
        linked_order_id=row.get("linked_order_id") or None,
        expected_category=(row.get("expected_category") or None),
        source_dataset=row.get("source_dataset") or None,
        source_ticket_ref=row.get("source_ticket_ref") or None,
        source_language=row.get("source_language") or None,
        source_queue=row.get("source_queue") or None,
        source_subject=row.get("source_subject") or None,
        source_body=row.get("source_body") or None,
    )


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


def build_template(project_root: Path, sample_size: int, seed: int, output_path: Path) -> dict[str, Any]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides={"MODEL_BACKEND": "rule"})
    rows = _load_seed_tickets(project_root / "data" / "seed" / "tickets.csv")
    sampled_rows = _balanced_sample(rows, sample_size=sample_size, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(sampled_rows, start=1):
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

        records.append(
            {
                "sample_id": f"GOLD-{index:04d}",
                "annotation_status": "pending",
                "annotator": "",
                "review_round": "",
                "notes": "",
                "ticket_id": ticket.ticket_id,
                "customer_id": ticket.customer_id,
                "channel": ticket.channel,
                "customer_tier": ticket.customer_tier,
                "product": ticket.product,
                "created_at": ticket.created_at.isoformat(),
                "linked_order_id": ticket.linked_order_id or "",
                "source_dataset": ticket.source_dataset or "",
                "source_ticket_ref": ticket.source_ticket_ref or "",
                "source_language": ticket.source_language or "",
                "source_queue": ticket.source_queue or "",
                "source_subject": ticket.source_subject or "",
                "source_body": ticket.source_body or "",
                "mapped_title_cn": ticket.title,
                "mapped_body_cn": ticket.body,
                "silver_category": triage.category,
                "silver_priority": triage.priority,
                "silver_sla_risk": str(triage.sla_risk).lower(),
                "silver_action_type": action.action_type,
                "silver_approval_required": str(action.requires_approval).lower(),
                "silver_target_status": str(getattr(action, "target_status", "") or _target_status_from_action(action.action_type, action.tool_args)),
                "gold_category": "",
                "gold_priority": "",
                "gold_sla_risk": "",
                "gold_action_type": "",
                "gold_approval_required": "",
                "gold_target_status": "",
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    return {
        "output_path": str(output_path),
        "sample_size": len(records),
        "source_dataset_breakdown": _breakdown(records, "source_dataset"),
        "category_hint_breakdown": _breakdown(records, "silver_category"),
    }


def _breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counter[key] = counter.get(key, 0) + 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a public-ticket structured gold annotation template.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Default: <project_root>/data/gold_eval/public_structured_gold_template.csv",
    )
    args = parser.parse_args()

    output_path = args.output_path or (args.project_root / "data" / "gold_eval" / "public_structured_gold_template.csv")
    summary = build_template(args.project_root, sample_size=args.sample_size, seed=args.seed, output_path=output_path)
    print(f"Structured gold template written to: {summary['output_path']}")
    print(f"Sample size: {summary['sample_size']}")
    print(f"Source datasets: {summary['source_dataset_breakdown']}")
    print(f"Silver category hints: {summary['category_hint_breakdown']}")


if __name__ == "__main__":
    main()
