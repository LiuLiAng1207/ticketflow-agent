from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import TicketFlowSettings
from .db import TicketFlowRepository


OUTPUT_FIELDS = [
    "sample_id",
    "annotation_status",
    "annotator",
    "notes",
    "ticket_id",
    "gold_category",
    "gold_priority",
    "gold_sla_risk",
    "gold_action_type",
    "gold_target_status",
    "gold_requires_policy",
    "gold_policy_doc_id",
    "gold_requires_history",
    "gold_history_doc_id",
    "gold_should_fallback_without_evidence",
    "gold_key_claims",
    "selection_reason",
    "hard_case_tags",
    "customer_tier",
    "linked_order_id",
    "mapped_title_cn",
    "mapped_body_cn",
]

PRIORITY_ORDER = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
CATEGORY_TARGETS = {
    "billing_refund": 12,
    "delivery_issue": 10,
    "technical_issue": 10,
    "account_access": 4,
    "general_inquiry": 4,
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _split_tags(value: str | None) -> list[str]:
    return [tag for tag in str(value or "").split("|") if tag]


def _row_score(row: dict[str, str]) -> tuple[int, int, int, str]:
    tags = _split_tags(row.get("hard_case_tags"))
    requires_policy = int(_requires_policy(row))
    requires_history = int(_requires_history(row))
    priority = PRIORITY_ORDER.get((row.get("gold_priority") or "low").strip(), 0)
    return (
        requires_policy + requires_history,
        len(tags),
        priority,
        row["ticket_id"],
    )


def _requires_policy(row: dict[str, str]) -> bool:
    category = (row.get("gold_category") or "").strip()
    action = (row.get("gold_action_type") or "").strip()
    tags = set(_split_tags(row.get("hard_case_tags")))
    if action in {"refund", "escalation", "sla_override", "close_ticket_without_contact"}:
        return True
    if category == "billing_refund":
        return True
    return bool(tags & {"refund_boundary", "approval_boundary", "outage_candidate"})


def _requires_history(row: dict[str, str]) -> bool:
    category = (row.get("gold_category") or "").strip()
    tags = set(_split_tags(row.get("hard_case_tags")))
    if category in {"delivery_issue", "technical_issue", "account_access"}:
        return True
    if category == "general_inquiry":
        return True
    return bool(tags & {"refund_boundary", "high_no_sla_candidate", "time_pressure"})


def _should_fallback_without_evidence(row: dict[str, str]) -> bool:
    action = (row.get("gold_action_type") or "").strip()
    tags = set(_split_tags(row.get("hard_case_tags")))
    if action in {"refund", "escalation"}:
        return True
    return bool(tags & {"refund_boundary", "outage_candidate", "approval_boundary"})


def _claim_hints(row: dict[str, str]) -> str:
    action = (row.get("gold_action_type") or "").strip()
    if action == "refund":
        return "退款|审核|进展"
    if action == "escalation":
        return "升级|优先排查|进展"
    if action == "request_info":
        return "补充|订单号|截图"
    if action == "troubleshoot":
        return "排查|报错|影响范围"
    if action == "status_update":
        return "同步|进展|处理"
    return "处理|进展"


def _selection_reason(row: dict[str, str]) -> str:
    parts: list[str] = []
    if _requires_policy(row):
        parts.append("policy_sensitive")
    if _requires_history(row):
        parts.append("history_sensitive")
    if _should_fallback_without_evidence(row):
        parts.append("fallback_sensitive")
    tags = _split_tags(row.get("hard_case_tags"))
    if tags:
        parts.append(",".join(tags[:2]))
    return "|".join(parts)


def _pick_policy_doc_id(repository: TicketFlowRepository, row: dict[str, str]) -> str:
    if not _requires_policy(row):
        return ""
    action_type = (row.get("gold_action_type") or "").strip()
    category = (row.get("gold_category") or "").strip()
    if action_type == "request_info" and category == "billing_refund":
        action_type = "refund"
    elif action_type == "status_update" and category == "delivery_issue":
        action_type = "escalation"
    query = "\n".join(
        filter(
            None,
            [
                row.get("mapped_title_cn"),
                row.get("mapped_body_cn"),
                row.get("gold_category"),
                action_type,
            ],
        )
    )
    hits = repository.lookup_policy(action_type, query=query, limit=1)
    return str(hits[0]["policy_id"]) if hits else ""


def _pick_history_doc_id(repository: TicketFlowRepository, row: dict[str, str]) -> str:
    if not _requires_history(row):
        return ""
    history = repository.get_ticket_history(row["ticket_id"], limit=1)
    if history:
        return str(history[0]["event_id"])
    query = "\n".join(filter(None, [row.get("mapped_title_cn"), row.get("mapped_body_cn")]))
    hits = repository.search_related_history(query=query, limit=1)
    return str(hits[0]["event_id"]) if hits else ""


def build_rag_sensitive_eval(
    *,
    project_root: Path,
    canonical_csv: Path,
    output_path: Path,
) -> dict[str, Any]:
    settings = TicketFlowSettings.from_project_root(project_root)
    repository = TicketFlowRepository(settings.db_path)
    repository.bootstrap(settings.seed_dir)

    rows = [row for row in _load_csv(canonical_csv) if (row.get("gold_split") or "").strip() == "eval"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["gold_category"]].append(row)

    selected: list[dict[str, str]] = []
    seen_ticket_ids: set[str] = set()
    for category, target in CATEGORY_TARGETS.items():
        candidates = sorted(grouped.get(category, []), key=_row_score, reverse=True)
        for row in candidates:
            if len([item for item in selected if item["gold_category"] == category]) >= target:
                break
            if row["ticket_id"] in seen_ticket_ids:
                continue
            selected.append(row)
            seen_ticket_ids.add(row["ticket_id"])

    if len(selected) < sum(CATEGORY_TARGETS.values()):
        remainder = sorted(rows, key=_row_score, reverse=True)
        for row in remainder:
            if row["ticket_id"] in seen_ticket_ids:
                continue
            selected.append(row)
            seen_ticket_ids.add(row["ticket_id"])
            if len(selected) >= sum(CATEGORY_TARGETS.values()):
                break

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        output_rows.append(
            {
                "sample_id": f"RAGS-{index:04d}",
                "annotation_status": "bootstrap",
                "annotator": "codex_rag_eval_v1",
                "notes": "rag_sensitive_bootstrap_v1_from_canonical_eval",
                "ticket_id": row["ticket_id"],
                "gold_category": row["gold_category"],
                "gold_priority": row["gold_priority"],
                "gold_sla_risk": row["gold_sla_risk"],
                "gold_action_type": row["gold_action_type"],
                "gold_target_status": row["gold_target_status"],
                "gold_requires_policy": str(_requires_policy(row)).lower(),
                "gold_policy_doc_id": _pick_policy_doc_id(repository, row),
                "gold_requires_history": str(_requires_history(row)).lower(),
                "gold_history_doc_id": _pick_history_doc_id(repository, row),
                "gold_should_fallback_without_evidence": str(_should_fallback_without_evidence(row)).lower(),
                "gold_key_claims": _claim_hints(row),
                "selection_reason": _selection_reason(row),
                "hard_case_tags": row.get("hard_case_tags") or "",
                "customer_tier": row.get("customer_tier") or "",
                "linked_order_id": row.get("linked_order_id") or "",
                "mapped_title_cn": row.get("mapped_title_cn") or "",
                "mapped_body_cn": row.get("mapped_body_cn") or "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    by_category: dict[str, int] = defaultdict(int)
    policy_required = 0
    history_required = 0
    fallback_required = 0
    for row in output_rows:
        by_category[row["gold_category"]] += 1
        policy_required += int(_truthy(row["gold_requires_policy"]))
        history_required += int(_truthy(row["gold_requires_history"]))
        fallback_required += int(_truthy(row["gold_should_fallback_without_evidence"]))

    return {
        "output_path": str(output_path),
        "sample_size": len(output_rows),
        "category_breakdown": dict(sorted(by_category.items())),
        "policy_required_count": policy_required,
        "history_required_count": history_required,
        "fallback_required_count": fallback_required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small RAG-sensitive evaluation CSV from canonical structured gold.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--canonical-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "gold_eval" / "public_structured_gold_canonical_v3.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "gold_eval" / "rag_sensitive_eval_v1.csv",
    )
    args = parser.parse_args()
    summary = build_rag_sensitive_eval(
        project_root=args.project_root,
        canonical_csv=args.canonical_csv,
        output_path=args.output_path,
    )
    print(summary)


if __name__ == "__main__":
    main()
