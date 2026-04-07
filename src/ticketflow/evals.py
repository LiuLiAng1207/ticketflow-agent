from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .graph import TicketFlowRunner
from .models import ActionProposal, ReviewDecision, TicketRecord, TicketResponse


CATEGORY_LABELS = [
    "billing_refund",
    "delivery_issue",
    "technical_issue",
    "account_access",
    "general_inquiry",
]

ACTION_ALIGNMENT_KEYWORDS = {
    "refund": ["退款", "审核", "财务", "退回"],
    "escalation": ["升级", "专家团队", "专项处理", "值班", "继续推进"],
    "request_info": ["补充", "订单号", "截图", "凭证", "信息"],
    "status_update": ["核验", "同步", "跟进", "处理", "进展"],
    "troubleshoot": ["排查", "日志", "报错", "影响范围", "绕过"],
    "sla_override": ["sla", "特批", "人工确认"],
    "close_ticket_without_contact": ["人工确认", "联系客户"],
}

FORBIDDEN_REPLY_CLAIMS = {
    "refund_completed": ["退款成功", "已经退款", "已完成退款", "款项已退回"],
    "resolved": ["问题已解决", "已经修复", "故障已恢复", "已经处理完成"],
    "hard_commitment": ["保证", "一定会", "明天上线", "今天上线", "24小时内", "立刻到账"],
}


def _expected_escalation(ticket: TicketRecord, order) -> bool:
    if ticket.expected_category == "technical_issue" and ticket.customer_tier == "enterprise":
        return True
    if ticket.expected_category == "delivery_issue" and order is not None:
        return order.status == "pending" or (order.delivered_days_ago or 0) > 5
    return False


def _expected_sla_risk(ticket: TicketRecord) -> bool:
    text = f"{ticket.title}\n{ticket.body}".lower()
    if ticket.expected_category != "technical_issue":
        return False
    return any(keyword in text for keyword in ["sla", "生产", "影响生产", "sev1", "紧急", "critical"])


def _expected_action(ticket: TicketRecord, order) -> tuple[str, bool, str]:
    category = ticket.expected_category or "general_inquiry"
    if category == "billing_refund":
        if not ticket.linked_order_id or order is None:
            return "request_info", False, "waiting_on_customer"
        if order.eligible_for_refund:
            return "refund", True, "pending_finance"
        return "status_update", False, "in_progress"
    if category == "technical_issue":
        if ticket.customer_tier == "enterprise" or _expected_sla_risk(ticket):
            return "escalation", True, "escalated"
        return "troubleshoot", False, "investigating"
    if category == "delivery_issue":
        if order is not None and (order.status == "pending" or (order.delivered_days_ago or 0) > 5):
            return "escalation", True, "escalated"
        return "status_update", False, "monitoring"
    if category == "account_access":
        return "troubleshoot", False, "investigating"
    return "status_update", False, "in_progress"


def _binary_f1(predictions: list[bool], expected: list[bool]) -> tuple[float, float, float]:
    tp = sum(1 for pred, exp in zip(predictions, expected) if pred and exp)
    fp = sum(1 for pred, exp in zip(predictions, expected) if pred and not exp)
    fn = sum(1 for pred, exp in zip(predictions, expected) if not pred and exp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def _macro_f1(predictions: list[str], expected: list[str], labels: list[str]) -> float:
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for pred, exp in zip(predictions, expected) if pred == label and exp == label)
        fp = sum(1 for pred, exp in zip(predictions, expected) if pred == label and exp != label)
        fn = sum(1 for pred, exp in zip(predictions, expected) if pred != label and exp == label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_scores.append(f1)
    return sum(f1_scores) / max(len(f1_scores), 1)


def _recall(predictions: list[bool], expected: list[bool]) -> float:
    positives = sum(1 for exp in expected if exp)
    true_positives = sum(1 for pred, exp in zip(predictions, expected) if pred and exp)
    return true_positives / max(positives, 1)


def _hit_source(docs: list[Any], source_type: str) -> bool:
    for doc in docs:
        if isinstance(doc, dict):
            value = doc.get("source_type")
        else:
            value = getattr(doc, "source_type", None)
        if value == source_type:
            return True
    return False


def _score_rag_coverage(runner: TicketFlowRunner, tickets: list[TicketRecord]) -> dict[str, Any]:
    current_policy_hits = 0
    current_kb_hits = 0
    current_history_hits = 0
    baseline_policy_hits = 0
    baseline_kb_hits = 0
    baseline_history_hits = 0
    vector_usage = 0

    for ticket in tickets:
        current = runner.retriever.retrieve(ticket)
        baseline = runner.retriever.baseline_retrieve(ticket)

        current_policy_hits += int(_hit_source(current.docs, "policy"))
        current_kb_hits += int(_hit_source(current.docs, "kb"))
        current_history_hits += int(_hit_source(current.docs, "history"))
        baseline_policy_hits += int(_hit_source(baseline.docs, "policy"))
        baseline_kb_hits += int(_hit_source(baseline.docs, "kb"))
        baseline_history_hits += int(_hit_source(baseline.docs, "history"))
        vector_usage += int(current.stats.used_vector)

    sample_size = max(len(tickets), 1)

    def _rate(count: int) -> float:
        return round(count / sample_size, 3)

    def _lift(current: float, baseline: float) -> float:
        if baseline == 0:
            return 1.0 if current > 0 else 0.0
        return round((current - baseline) / baseline, 3)

    current_policy_rate = _rate(current_policy_hits)
    current_kb_rate = _rate(current_kb_hits)
    current_history_rate = _rate(current_history_hits)
    baseline_policy_rate = _rate(baseline_policy_hits)
    baseline_kb_rate = _rate(baseline_kb_hits)
    baseline_history_rate = _rate(baseline_history_hits)

    return {
        "rag_policy_hit_rate": current_policy_rate,
        "rag_kb_hit_rate": current_kb_rate,
        "rag_history_hit_rate": current_history_rate,
        "baseline_policy_hit_rate": baseline_policy_rate,
        "baseline_kb_hit_rate": baseline_kb_rate,
        "baseline_history_hit_rate": baseline_history_rate,
        "policy_hit_lift_pct": _lift(current_policy_rate, baseline_policy_rate),
        "kb_hit_lift_pct": _lift(current_kb_rate, baseline_kb_rate),
        "history_hit_lift_pct": _lift(current_history_rate, baseline_history_rate),
        "vector_retrieval_usage_rate": _rate(vector_usage),
    }


def _predicted_target_status(ticket: TicketRecord, action: ActionProposal) -> str:
    if getattr(action, "target_status", None):
        return str(action.target_status)
    if action.action_type == "refund":
        return "pending_finance"
    if action.action_type == "escalation":
        return "escalated"
    if action.action_type == "request_info":
        return "waiting_on_customer"
    if action.action_type == "troubleshoot":
        return "investigating"
    if action.action_type == "status_update":
        return action.tool_args.get("status") or ("monitoring" if ticket.expected_category == "delivery_issue" else "in_progress")
    if action.action_type == "sla_override":
        return "in_progress"
    return "pending_human"


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _reply_alignment_ok(reply: str, action_type: str) -> bool:
    keywords = ACTION_ALIGNMENT_KEYWORDS.get(action_type, [])
    return _contains_any(reply, keywords) if keywords else True


def _wrong_order_reference(reply: str, ticket: TicketRecord) -> bool:
    order_ids = set(re.findall(r"ORD-\d{4}", reply, flags=re.IGNORECASE))
    if not order_ids:
        return False
    expected = (ticket.linked_order_id or "").upper()
    return any(order_id.upper() != expected for order_id in order_ids)


def _reply_policy_violation(reply: str, ticket: TicketRecord, action: ActionProposal, execution_result: dict[str, Any]) -> bool:
    del ticket
    if action.action_type == "request_info":
        return not _contains_any(reply, ACTION_ALIGNMENT_KEYWORDS["request_info"])
    if action.requires_approval and _contains_any(reply, FORBIDDEN_REPLY_CLAIMS["refund_completed"]):
        return True
    if action.action_type == "status_update" and _contains_any(reply, FORBIDDEN_REPLY_CLAIMS["hard_commitment"]):
        return True
    if execution_result.get("status") != "executed" and _contains_any(reply, FORBIDDEN_REPLY_CLAIMS["resolved"]):
        return True
    return False


def _reply_hallucination(reply: str, ticket: TicketRecord, action: ActionProposal, execution_result: dict[str, Any]) -> bool:
    if _wrong_order_reference(reply, ticket):
        return True
    if action.action_type != "refund" and _contains_any(reply, FORBIDDEN_REPLY_CLAIMS["refund_completed"]):
        return True
    if execution_result.get("status") != "executed" and _contains_any(reply, FORBIDDEN_REPLY_CLAIMS["resolved"]):
        return True
    if _contains_any(reply, ["已经发货", "已送达"]) and action.action_type != "status_update":
        return True
    return False


def _reply_grounded(response: TicketResponse, ticket: TicketRecord, action: ActionProposal, execution_result: dict[str, Any]) -> bool:
    if response.review_grounded is not None:
        return bool(response.review_grounded)
    if _reply_hallucination(response.customer_reply, ticket, action, execution_result):
        return False
    if _reply_policy_violation(response.customer_reply, ticket, action, execution_result):
        return False
    if not _reply_alignment_ok(response.customer_reply, action.action_type):
        return False
    if action.action_type == "request_info":
        return True
    return bool(response.citations)


def _reply_policy_violation_flag(
    response: TicketResponse,
    ticket: TicketRecord,
    action: ActionProposal,
    execution_result: dict[str, Any],
) -> bool:
    if response.review_policy_safe is not None:
        return not bool(response.review_policy_safe)
    return _reply_policy_violation(response.customer_reply, ticket, action, execution_result)


def _reply_hallucination_flag(
    response: TicketResponse,
    ticket: TicketRecord,
    action: ActionProposal,
    execution_result: dict[str, Any],
) -> bool:
    if response.review_hallucination_risk is not None:
        return bool(response.review_hallucination_risk)
    return _reply_hallucination(response.customer_reply, ticket, action, execution_result)


def _reply_supported_claim_ratio(
    response: TicketResponse,
    ticket: TicketRecord,
    action: ActionProposal,
    execution_result: dict[str, Any],
) -> float:
    if response.review_supported_claim_ratio is not None:
        return float(response.review_supported_claim_ratio)
    grounded = _reply_grounded(response, ticket, action, execution_result)
    policy_safe = not _reply_policy_violation(response.customer_reply, ticket, action, execution_result)
    hallucination = _reply_hallucination(response.customer_reply, ticket, action, execution_result)
    if grounded and policy_safe and not hallucination:
        return 1.0
    if grounded and not hallucination:
        return 0.75
    if not grounded and not hallucination:
        return 0.5
    return 0.25


def _reply_clarity_score(reply: str) -> float:
    text = reply.strip()
    length = len(text)
    sentence_count = max(text.count("。") + text.count("！") + text.count("？"), 1)
    score = 1.0
    if length < 18 or length > 120:
        score -= 0.2
    if sentence_count > 3:
        score -= 0.2
    if "  " in text or "\n\n" in text:
        score -= 0.1
    return max(score, 0.0)


def _reply_rubric_proxy_score(
    response: TicketResponse,
    ticket: TicketRecord,
    action: ActionProposal,
    execution_result: dict[str, Any],
) -> float:
    alignment = 1.0 if _reply_alignment_ok(response.customer_reply, action.action_type) else 0.0
    groundedness = 1.0 if _reply_grounded(response, ticket, action, execution_result) else 0.0
    compliance = 1.0 if not _reply_policy_violation_flag(response, ticket, action, execution_result) else 0.0
    non_hallucinated = 1.0 if not _reply_hallucination_flag(response, ticket, action, execution_result) else 0.0
    clarity = _reply_clarity_score(response.customer_reply)
    return round(((alignment + groundedness + compliance + non_hallucinated + clarity) / 5) * 5, 3)


def run_evaluation(
    project_root: Path,
    sample_size: int = 30,
    overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides=overrides)
    runner.reset_demo_data()
    tickets = runner.list_open_tickets(limit=sample_size)

    classification_correct = 0
    predicted_categories: list[str] = []
    expected_categories: list[str] = []
    predicted_sla_risks: list[bool] = []
    expected_sla_risks: list[bool] = []
    predicted_escalations: list[bool] = []
    expected_escalations: list[bool] = []
    action_type_correct = 0
    target_status_correct = 0
    predicted_approval_required: list[bool] = []
    expected_approval_required: list[bool] = []
    approval_sensitive = 0
    approval_interrupted = 0
    usable_replies = 0
    total_steps = 0
    triage_cloud = 0
    triage_minimind = 0
    action_cloud = 0
    action_minimind = 0
    reply_cloud = 0
    reply_minimind = 0
    fallback_cases = 0
    citation_grounded = 0
    unsupported_sensitive_actions = 0
    incident_expected = 0
    incident_success = 0
    kb_expected = 0
    kb_success = 0
    external_triggered = 0
    external_nonblocking_ok = 0
    external_latency_total = 0
    external_latency_count = 0
    grounded_replies = 0
    policy_violation_count = 0
    hallucination_count = 0
    reply_reviewed = 0
    reply_review_passed = 0
    reply_supported_claim_ratios: list[float] = []
    reply_rubric_scores: list[float] = []
    bad_cases: list[dict[str, object]] = []

    rag_metrics = _score_rag_coverage(runner, tickets)

    for ticket in tickets:
        initial = runner.run_ticket(ticket.ticket_id, thread_id=f"eval-{ticket.ticket_id}")
        state = initial.state
        triage_result = state["triage_result"]

        expected_category = ticket.expected_category or "general_inquiry"
        expected_categories.append(expected_category)
        predicted_categories.append(triage_result.category)
        expected_sla = _expected_sla_risk(ticket)
        expected_sla_risks.append(expected_sla)
        predicted_sla_risks.append(triage_result.sla_risk)

        if triage_result.category == expected_category:
            classification_correct += 1
        else:
            bad_cases.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "kind": "classification_mismatch",
                    "expected": expected_category,
                    "predicted": triage_result.category,
                    "title": ticket.title,
                }
            )

        if triage_result.decision_source == "llm_cloud":
            triage_cloud += 1
        elif triage_result.decision_source == "llm_minimind":
            triage_minimind += 1

        proposed_action = state["proposed_action"]
        if proposed_action.decision_source == "llm_cloud":
            action_cloud += 1
        elif proposed_action.decision_source == "llm_minimind":
            action_minimind += 1
        order = runner.repository.get_order_status(ticket.linked_order_id)
        expected_action_type, expected_requires_approval, expected_target_status = _expected_action(ticket, order)

        action_type_correct += int(proposed_action.action_type == expected_action_type)
        predicted_approval_required.append(proposed_action.requires_approval)
        expected_approval_required.append(expected_requires_approval)
        predicted_target_status = _predicted_target_status(ticket, proposed_action)
        target_status_correct += int(predicted_target_status == expected_target_status)

        is_sensitive = proposed_action.action_type in {"refund", "escalation", "sla_override", "close_ticket_without_contact"}
        if is_sensitive:
            approval_sensitive += 1
            if not state.get("policy_hits"):
                unsupported_sensitive_actions += 1

        if initial.interrupted:
            approval_interrupted += 1
            result = runner.resume_ticket(
                initial.state["thread_id"],
                ReviewDecision(decision="approve", comment="Auto-approved for offline evaluation"),
            )
            state = result.state

        draft_reply = state["draft_reply"]
        execution_result = state["execution_result"]
        if draft_reply.draft_source == "llm_cloud":
            reply_cloud += 1
        elif draft_reply.draft_source == "llm_minimind":
            reply_minimind += 1
        if draft_reply.customer_reply and draft_reply.internal_note:
            usable_replies += 1
        if draft_reply.citations:
            citation_grounded += 1

        grounded = _reply_grounded(draft_reply, ticket, proposed_action, execution_result)
        policy_violation = _reply_policy_violation_flag(draft_reply, ticket, proposed_action, execution_result)
        hallucination = _reply_hallucination_flag(draft_reply, ticket, proposed_action, execution_result)
        supported_claim_ratio = _reply_supported_claim_ratio(draft_reply, ticket, proposed_action, execution_result)
        grounded_replies += int(grounded)
        policy_violation_count += int(policy_violation)
        hallucination_count += int(hallucination)
        if draft_reply.review_passed is not None:
            reply_reviewed += 1
            reply_review_passed += int(bool(draft_reply.review_passed))
        reply_supported_claim_ratios.append(supported_claim_ratio)
        reply_rubric_scores.append(_reply_rubric_proxy_score(draft_reply, ticket, proposed_action, execution_result))

        if triage_result.fallback_reason or proposed_action.fallback_reason or draft_reply.fallback_reason:
            fallback_cases += 1

        expected_escalations.append(_expected_escalation(ticket, order))
        predicted_escalations.append(proposed_action.action_type == "escalation")
        total_steps += len(state.get("trace", []))

        external_ops = state.get("external_ops", [])
        if external_ops:
            external_triggered += 1
            if state["draft_reply"].status:
                external_nonblocking_ok += 1
        if proposed_action.action_type == "escalation" and state["execution_result"]["status"] == "executed":
            incident_expected += 1
            if any((op.status if hasattr(op, "status") else op.get("status")) == "sent" and (op.op_type if hasattr(op, "op_type") else op.get("op_type")) == "incident_email" for op in external_ops):
                incident_success += 1

        if runner._should_submit_kb_candidate(ticket, triage_result, state):
            kb_expected += 1
            if any((op.status if hasattr(op, "status") else op.get("status")) == "sent" and (op.op_type if hasattr(op, "op_type") else op.get("op_type")) == "kb_candidate_email" for op in external_ops):
                kb_success += 1

        for op in external_ops:
            latency = op.latency_ms if hasattr(op, "latency_ms") else op.get("latency_ms")
            if latency:
                external_latency_total += latency
                external_latency_count += 1

    precision, recall, f1 = _binary_f1(predicted_escalations, expected_escalations)
    sla_precision, sla_recall, sla_f1 = _binary_f1(predicted_sla_risks, expected_sla_risks)
    sla_negatives = sum(1 for item in expected_sla_risks if not item)
    sla_false_positives = sum(1 for pred, gold in zip(predicted_sla_risks, expected_sla_risks) if pred and not gold)
    sample_size = max(len(tickets), 1)
    report = {
        "sample_size": len(tickets),
        "classification_accuracy": round(classification_correct / sample_size, 3),
        "category_macro_f1": round(_macro_f1(predicted_categories, expected_categories, CATEGORY_LABELS), 3),
        "sla_risk_precision": round(sla_precision, 3),
        "sla_risk_recall": round(sla_recall, 3),
        "sla_risk_f1": round(sla_f1, 3),
        "sla_risk_false_positive_rate": round(sla_false_positives / max(sla_negatives, 1), 3),
        "action_type_accuracy": round(action_type_correct / sample_size, 3),
        "approval_required_recall": round(_recall(predicted_approval_required, expected_approval_required), 3),
        "target_status_accuracy": round(target_status_correct / sample_size, 3),
        "escalation_precision": round(precision, 3),
        "escalation_recall": round(recall, 3),
        "escalation_f1": round(f1, 3),
        "sensitive_action_approval_interception_rate": round(approval_interrupted / max(approval_sensitive, 1), 3),
        "first_reply_usable_rate": round(usable_replies / sample_size, 3),
        "groundedness_rate": round(grounded_replies / sample_size, 3),
        "policy_violation_rate": round(policy_violation_count / sample_size, 3),
        "hallucination_rate": round(hallucination_count / sample_size, 3),
        "reply_reviewed_rate": round(reply_reviewed / sample_size, 3),
        "reply_review_pass_rate": round(reply_review_passed / max(reply_reviewed, 1), 3),
        "reply_supported_claim_ratio_mean": round(sum(reply_supported_claim_ratios) / max(len(reply_supported_claim_ratios), 1), 3),
        "reply_rubric_proxy_mean": round(sum(reply_rubric_scores) / max(len(reply_rubric_scores), 1), 3),
        "average_processing_steps": round(total_steps / sample_size, 2),
        "triage_cloud_rate": round(triage_cloud / sample_size, 3),
        "triage_minimind_rate": round(triage_minimind / sample_size, 3),
        "action_cloud_rate": round(action_cloud / sample_size, 3),
        "action_minimind_rate": round(action_minimind / sample_size, 3),
        "reply_cloud_rate": round(reply_cloud / sample_size, 3),
        "reply_minimind_rate": round(reply_minimind / sample_size, 3),
        "fallback_case_rate": round(fallback_cases / sample_size, 3),
        "citation_grounding_rate": round(citation_grounded / sample_size, 3),
        "unsupported_sensitive_action_rate": round(unsupported_sensitive_actions / max(approval_sensitive, 1), 3),
        "incident_email_trigger_rate": round(incident_expected / sample_size, 3),
        "incident_email_success_rate": round(incident_success / max(incident_expected, 1), 3),
        "kb_candidate_trigger_rate": round(kb_expected / sample_size, 3),
        "kb_candidate_success_rate": round(kb_success / max(kb_expected, 1), 3),
        "external_nonblocking_recovery_rate": round(external_nonblocking_ok / max(external_triggered, 1), 3),
        "external_delivery_avg_latency_ms": round(external_latency_total / max(external_latency_count, 1), 1),
        "bad_case_examples": bad_cases[:5],
    }
    report.update(rag_metrics)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TicketFlow offline evaluation.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TicketFlow project root.",
    )
    parser.add_argument(
        "--model-backend",
        choices=["rule", "cloud_api", "minimind_api", "minimind_split"],
        default="rule",
        help="离线评测使用的模型后端。",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()
    report = run_evaluation(
        args.project_root,
        sample_size=args.sample_size,
        overrides={"MODEL_BACKEND": args.model_backend},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
