from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "sample_id",
    "annotation_status",
    "annotator",
    "review_round",
    "notes",
    "gold_split",
    "hard_case_tags",
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

PRIORITY_ORDER = {"urgent": 4, "high": 3, "medium": 2, "low": 1}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _contains(text: str, *keywords: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _hard_case_tags(row: dict[str, str]) -> list[str]:
    text = f"{row.get('title', '')}\n{row.get('body', '')}\n{row.get('source_subject', '')}\n{row.get('source_body', '')}"
    tags: list[str] = []
    if row.get("customer_tier") == "enterprise":
        tags.append("enterprise")
    if row.get("expected_category") == "technical_issue" and _contains(text, "生产", "outage", "incident", "crash", "宕机", "SLA"):
        tags.append("outage_candidate")
    if row.get("expected_category") == "billing_refund" and (not row.get("linked_order_id") or _contains(text, "补充", "截图", "找不到订单", "need additional information")):
        tags.append("refund_boundary")
    if row.get("expected_category") in {"technical_issue", "account_access"} and _contains(text, "影响使用", "影响业务", "多人", "team", "blocked") and not _contains(text, "SLA", "生产", "production"):
        tags.append("high_no_sla_candidate")
    if row.get("expected_category") in {"delivery_issue", "billing_refund"} or _contains(text, "审批", "escalate", "urgent follow-up"):
        tags.append("approval_boundary")
    if _contains(text, "尽快", "as soon as possible", "prompt solution", "urgent") and not tags:
        tags.append("time_pressure")
    return sorted(set(tags))


def _bootstrap_labels(row: dict[str, str]) -> dict[str, str]:
    category = (row.get("expected_category") or "general_inquiry").strip()
    text = f"{row.get('title', '')}\n{row.get('body', '')}\n{row.get('source_subject', '')}\n{row.get('source_body', '')}"
    enterprise = row.get("customer_tier") == "enterprise"
    has_order = bool((row.get("linked_order_id") or "").strip())

    if category == "technical_issue":
        sla_risk = _contains(text, "SLA", "生产", "production", "contract", "违约")
        if sla_risk:
            priority = "urgent"
        elif enterprise or _contains(text, "影响业务", "blocked", "many users", "team", "严重", "故障", "error"):
            priority = "high"
        else:
            priority = "medium"
        action_type = "escalation" if priority in {"high", "urgent"} else "troubleshoot"
    elif category == "billing_refund":
        sla_risk = False
        if _contains(text, "double charge", "重复收费", "incorrect charging", "financial impact"):
            priority = "high"
        else:
            priority = "medium"
        action_type = "refund" if has_order and not _contains(text, "补充", "截图", "找不到订单", "need any additional information") else "request_info"
    elif category == "delivery_issue":
        sla_risk = False
        if enterprise or _contains(text, "多日", "many days", "tracking has not updated", "earliest convenience", "尽快", "urgent"):
            priority = "high"
        else:
            priority = "medium"
        action_type = "escalation" if priority == "high" else "status_update"
    elif category == "account_access":
        sla_risk = False
        if enterprise or _contains(text, "影响使用", "影响业务", "cannot work", "blocked", "daily operations"):
            priority = "high"
        else:
            priority = "medium"
        action_type = "troubleshoot"
    else:
        sla_risk = False
        priority = "medium" if _contains(text, "尽快", "follow-up", "progress", "clarification") else "low"
        action_type = "status_update"

    requires_approval = action_type in {"refund", "escalation"}
    if action_type == "refund":
        target_status = "pending_finance"
    elif action_type == "escalation":
        target_status = "escalated"
    elif action_type == "request_info":
        target_status = "waiting_on_customer"
    elif action_type == "troubleshoot":
        target_status = "investigating"
    else:
        target_status = "monitoring" if category == "delivery_issue" else "in_progress"

    return {
        "gold_category": category,
        "gold_priority": priority,
        "gold_sla_risk": str(sla_risk).lower(),
        "gold_action_type": action_type,
        "gold_approval_required": str(requires_approval).lower(),
        "gold_target_status": target_status,
    }


def _merge_existing_annotations(paths: list[Path]) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in _load_csv(path):
            ticket_id = (row.get("ticket_id") or "").strip()
            if not ticket_id:
                continue
            merged[ticket_id] = row
    return merged


def _assign_split(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["gold_category"]].append(row)

    for values in grouped.values():
        values.sort(
            key=lambda item: (
                -len(item["hard_case_tags"].split("|") if item["hard_case_tags"] else []),
                -PRIORITY_ORDER.get(item["gold_priority"], 0),
                item["ticket_id"],
            )
        )
        for index, row in enumerate(values):
            if index < 16:
                row["gold_split"] = "eval"
            elif index < 24:
                row["gold_split"] = "dev"
            else:
                row["gold_split"] = "train"


def build_canonical_gold(
    *,
    project_root: Path,
    existing_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    tickets = _load_csv(project_root / "data" / "seed" / "tickets.csv")
    existing = _merge_existing_annotations(existing_paths)

    rows: list[dict[str, Any]] = []
    for index, ticket in enumerate(tickets, start=1):
        existing_row = existing.get(ticket["ticket_id"])
        bootstrap = _bootstrap_labels(ticket)
        hard_case_tags = "|".join(_hard_case_tags(ticket))

        base = {
            "sample_id": f"CGOLD-{index:04d}",
            "annotation_status": "final",
            "annotator": "codex_gold_v3",
            "review_round": "1",
            "notes": "canonical_public_ticket_gold_v3",
            "gold_split": "",
            "hard_case_tags": hard_case_tags,
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "channel": ticket["channel"],
            "customer_tier": ticket["customer_tier"],
            "product": ticket["product"],
            "created_at": ticket["created_at"],
            "linked_order_id": ticket.get("linked_order_id") or "",
            "source_dataset": ticket.get("source_dataset") or "",
            "source_ticket_ref": ticket.get("source_ticket_ref") or "",
            "source_language": ticket.get("source_language") or "",
            "source_queue": ticket.get("source_queue") or "",
            "source_subject": ticket.get("source_subject") or "",
            "source_body": ticket.get("source_body") or "",
            "mapped_title_cn": ticket["title"],
            "mapped_body_cn": ticket["body"],
            "silver_category": ticket.get("expected_category") or bootstrap["gold_category"],
            "silver_priority": bootstrap["gold_priority"],
            "silver_sla_risk": bootstrap["gold_sla_risk"],
            "silver_action_type": bootstrap["gold_action_type"],
            "silver_approval_required": bootstrap["gold_approval_required"],
            "silver_target_status": bootstrap["gold_target_status"],
            **bootstrap,
        }

        if existing_row is not None:
            for key in (
                "annotation_status",
                "annotator",
                "review_round",
                "notes",
                "gold_category",
                "gold_priority",
                "gold_sla_risk",
                "gold_action_type",
                "gold_approval_required",
                "gold_target_status",
            ):
                if str(existing_row.get(key) or "").strip():
                    base[key] = existing_row[key]
            if not base["notes"]:
                base["notes"] = "canonical_public_ticket_gold_v3_existing"

        rows.append(base)

    _assign_split(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    split_breakdown: dict[str, int] = defaultdict(int)
    hard_case_breakdown: dict[str, int] = defaultdict(int)
    for row in rows:
        split_breakdown[row["gold_split"]] += 1
        for tag in filter(None, str(row["hard_case_tags"]).split("|")):
            hard_case_breakdown[tag] += 1

    return {
        "output_path": str(output_path),
        "sample_size": len(rows),
        "split_breakdown": dict(sorted(split_breakdown.items())),
        "hard_case_breakdown": dict(sorted(hard_case_breakdown.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical 360-row structured gold CSV from seed tickets.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--existing-csv",
        action="append",
        type=Path,
        default=[
            Path(r"D:\Study\shixi\public_structured_gold_annotated_v1.csv"),
            Path(r"D:\Study\shixi\public_structured_gold_annotated_v2.csv"),
        ],
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(r"D:\Study\shixi\public_structured_gold_canonical_v3.csv"),
    )
    args = parser.parse_args()
    summary = build_canonical_gold(
        project_root=args.project_root,
        existing_paths=args.existing_csv,
        output_path=args.output_path,
    )
    print(summary)


if __name__ == "__main__":
    main()
