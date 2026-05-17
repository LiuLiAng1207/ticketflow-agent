from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import TicketFlowSettings
from .db import TicketFlowRepository
from .models import OrderRecord, TicketRecord
from .seed import build_seed_dataset


OUTPUT_FIELDS = [
    "sample_id",
    "ticket_id",
    "case_type",
    "gold_category",
    "gold_action_type",
    "gold_target_status",
    "gold_requires_approval",
    "gold_sufficient",
    "gold_should_fallback_without_evidence",
    "gold_evidence_type",
    "gold_entities_json",
    "attachment_filename",
    "attachment_file_type",
    "attachment_source_dataset",
    "attachment_ocr_text",
    "attachment_visual_summary",
    "notes",
]


CASE_TYPES = ["payment_verified", "payment_unverified", "error_screenshot", "account_screenshot", "document_form"]


def _truth(value: bool) -> str:
    return str(value).lower()


def _expected_action(ticket: TicketRecord, order: OrderRecord | None) -> tuple[str, str, bool]:
    category = ticket.expected_category or "general_inquiry"
    if category == "billing_refund":
        if not ticket.linked_order_id or order is None:
            return "request_info", "waiting_on_customer", False
        if order.eligible_for_refund:
            return "refund", "pending_finance", True
        return "status_update", "in_progress", False
    if category == "technical_issue":
        if ticket.customer_tier == "enterprise":
            return "escalation", "escalated", True
        return "troubleshoot", "investigating", False
    if category == "account_access":
        return "troubleshoot", "investigating", False
    if category == "delivery_issue":
        return "status_update", "monitoring", False
    return "status_update", "in_progress", False


def _candidate_case_type(ticket: TicketRecord, order: OrderRecord | None) -> str | None:
    if ticket.expected_category == "billing_refund" and ticket.linked_order_id and order is not None:
        return "payment_verified"
    if ticket.expected_category == "billing_refund" and not ticket.linked_order_id:
        return "payment_unverified"
    if ticket.expected_category == "technical_issue":
        return "error_screenshot"
    if ticket.expected_category == "account_access":
        return "account_screenshot"
    if ticket.expected_category == "general_inquiry":
        return "document_form"
    return None


def _attachment_payload(case_type: str, ticket: TicketRecord, order: OrderRecord | None, index: int) -> dict[str, Any]:
    if case_type == "payment_verified" and order is not None:
        return {
            "gold_evidence_type": "payment_screenshot",
            "gold_entities": {"order_id": order.order_id, "amount": float(order.amount)},
            "filename": "payment_success.png",
            "file_type": "image",
            "source_dataset": "synthetic_sroie_cord_style",
            "ocr_text": f"支付成功 订单号 {order.order_id} 金额 {order.amount} 元",
            "visual_summary": f"支付截图显示 {ticket.product} 已完成付款，金额 {order.amount} 元。",
            "notes": "SROIE/CORD-style receipt evidence; order system remains authoritative.",
        }
    if case_type == "payment_unverified":
        synthetic_order_id = f"ORD-UNVERIFIED-{index:04d}"
        return {
            "gold_evidence_type": "payment_screenshot",
            "gold_entities": {"order_id": synthetic_order_id, "amount": 299.0},
            "filename": "payment_claim_unverified.png",
            "file_type": "image",
            "source_dataset": "synthetic_sroie_cord_style",
            "ocr_text": f"支付成功 订单号 {synthetic_order_id} 金额 299.00 元",
            "visual_summary": "截图疑似显示支付成功，但订单系统尚未核验，不能单独放行退款。",
            "notes": "Attachment is auxiliary only; missing order source should trigger conservative fallback.",
        }
    if case_type == "error_screenshot":
        return {
            "gold_evidence_type": "error_screenshot",
            "gold_entities": {"error_code": "503"},
            "filename": "service_unavailable.png",
            "file_type": "image",
            "source_dataset": "synthetic_rico_style",
            "ocr_text": f"Error 503 Service Unavailable {ticket.product} 页面无法加载",
            "visual_summary": "报错截图显示页面服务不可用，页面中心有 503 错误提示。",
            "notes": "RICO-style UI screenshot with synthetic error overlay.",
        }
    if case_type == "account_screenshot":
        return {
            "gold_evidence_type": "error_screenshot",
            "gold_entities": {"error_code": "MFA-401"},
            "filename": "mfa_login_failed.png",
            "file_type": "image",
            "source_dataset": "synthetic_rico_style",
            "ocr_text": f"MFA-401 验证码校验失败 {ticket.product} 无法登录",
            "visual_summary": "登录截图显示 MFA 验证失败，用户无法进入控制台。",
            "notes": "RICO-style account access screenshot.",
        }
    return {
        "gold_evidence_type": "document_form",
        "gold_entities": {"processing_days": "2"},
        "filename": "support_form.pdf",
        "file_type": "pdf",
        "source_dataset": "synthetic_docvqa_funsd_style",
        "ocr_text": f"服务申请表 产品 {ticket.product} 处理时限 2 个工作日",
        "visual_summary": "PDF 表单包含产品名称、申请日期和处理时限字段。",
        "notes": "DocVQA/FUNSD-style form evidence.",
    }


def _targets(target_size: int) -> dict[str, int]:
    base = target_size // len(CASE_TYPES)
    targets = {case_type: base for case_type in CASE_TYPES}
    for case_type in CASE_TYPES[: target_size - base * len(CASE_TYPES)]:
        targets[case_type] += 1
    return targets


def build_multimodal_ticket_eval(*, project_root: Path, output_path: Path, target_size: int = 200) -> dict[str, Any]:
    settings = TicketFlowSettings.from_project_root(project_root)
    if not settings.seed_dir.exists() or not (settings.seed_dir / "tickets.csv").exists():
        build_seed_dataset(settings.seed_dir)
    repository = TicketFlowRepository(settings.db_path)
    repository.bootstrap(settings.seed_dir)

    grouped: dict[str, list[tuple[TicketRecord, OrderRecord | None]]] = defaultdict(list)
    for ticket in repository.list_open_tickets(limit=1000):
        order = repository.get_order_status(ticket.linked_order_id)
        case_type = _candidate_case_type(ticket, order)
        if case_type:
            grouped[case_type].append((ticket, order))

    selected: list[tuple[str, TicketRecord, OrderRecord | None]] = []
    seen: set[str] = set()
    for case_type, target in _targets(target_size).items():
        for ticket, order in grouped.get(case_type, []):
            if len([item for item in selected if item[0] == case_type]) >= target:
                break
            if ticket.ticket_id in seen:
                continue
            selected.append((case_type, ticket, order))
            seen.add(ticket.ticket_id)

    if len(selected) < target_size:
        for case_type in CASE_TYPES:
            for ticket, order in grouped.get(case_type, []):
                if len(selected) >= target_size:
                    break
                if ticket.ticket_id in seen:
                    continue
                selected.append((case_type, ticket, order))
                seen.add(ticket.ticket_id)
            if len(selected) >= target_size:
                break

    rows: list[dict[str, str]] = []
    for index, (case_type, ticket, order) in enumerate(selected, start=1):
        action_type, target_status, requires_approval = _expected_action(ticket, order)
        payload = _attachment_payload(case_type, ticket, order, index)
        expected_sufficient = not (case_type == "payment_unverified")
        rows.append(
            {
                "sample_id": f"MMT-{index:04d}",
                "ticket_id": ticket.ticket_id,
                "case_type": case_type,
                "gold_category": ticket.expected_category or "",
                "gold_action_type": action_type,
                "gold_target_status": target_status,
                "gold_requires_approval": _truth(requires_approval),
                "gold_sufficient": _truth(expected_sufficient),
                "gold_should_fallback_without_evidence": _truth(case_type == "payment_unverified"),
                "gold_evidence_type": payload["gold_evidence_type"],
                "gold_entities_json": json.dumps(payload["gold_entities"], ensure_ascii=False, sort_keys=True),
                "attachment_filename": payload["filename"],
                "attachment_file_type": payload["file_type"],
                "attachment_source_dataset": payload["source_dataset"],
                "attachment_ocr_text": payload["ocr_text"],
                "attachment_visual_summary": payload["visual_summary"],
                "notes": payload["notes"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_case_type = Counter(row["case_type"] for row in rows)
    summary = {
        "dataset": "multimodal_ticket_eval_v1",
        "sample_count": len(rows),
        "by_case_type": dict(by_case_type),
        "output_path": str(output_path),
        "notes": "Synthetic multimodal benchmark derived from public ticket data plus SROIE/CORD/RICO/DocVQA/FUNSD-style attachment evidence.",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TicketFlow multimodal ticket benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=200)
    args = parser.parse_args()
    output = args.output or args.project_root / "data" / "eval" / "multimodal_ticket_eval_v1.csv"
    summary = build_multimodal_ticket_eval(project_root=args.project_root, output_path=output, target_size=args.target_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
