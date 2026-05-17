from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .agents_llm import _revalidate_final_reply_against_contract
from .governance import build_action_route, get_requirement_profile, get_tool_approval_policy, infer_route_family
from .graph import TicketFlowRunner
from .models import ActionProposal, RetrievedDoc, TicketRecord, TriageResult


DEFAULT_CACHE_VERSION = "rag_sensitive_v2"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _priority_to_urgency(priority: str) -> str:
    mapping = {
        "low": "standard",
        "medium": "next_business_day",
        "high": "same_day",
        "urgent": "sev1",
    }
    return mapping.get(priority, "next_business_day")


def _split_claims(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def _split_eval_list(value: str | None) -> list[str]:
    normalized = str(value or "").replace("；", "|").replace("，", "|").replace(",", "|").replace(";", "|")
    return [item.strip() for item in normalized.split("|") if item.strip()]


def _row_route_family(row: dict[str, str]) -> str:
    explicit = (row.get("gold_route_family") or "").strip()
    if explicit:
        return explicit
    category = (row.get("gold_category") or "").strip()
    action_type = (row.get("gold_action_type") or "").strip()
    target_status = (row.get("gold_target_status") or "").strip()
    if category == "billing_refund" or action_type == "refund":
        return "refund_candidate"
    if action_type == "escalation" or target_status == "escalated":
        return "escalation_candidate"
    if category in {"technical_issue", "delivery_issue"} and target_status == "pending_human":
        return "escalation_candidate"
    return "standard_resolution"


def _required_source_types(row: dict[str, str]) -> set[str]:
    explicit = set(_split_eval_list(row.get("gold_required_sources")))
    if explicit:
        return explicit

    route_family = _row_route_family(row)
    if route_family in {"refund_candidate", "escalation_candidate", "standard_resolution"}:
        return set(get_requirement_profile(route_family).required_sources)

    required: set[str] = set()
    if _truthy(row.get("gold_requires_policy")):
        required.add("policy")
    if _truthy(row.get("gold_requires_history")):
        required.add("history")
    return required


def _available_source_types(docs: list[RetrievedDoc]) -> set[str]:
    return {doc.source_type for doc in docs if doc.source_type}


def _expected_sufficient(row: dict[str, str], available_sources: set[str]) -> bool:
    required_sources = _required_source_types(row)
    if not required_sources:
        return not str(row.get("gold_sufficient") or "").strip().lower() in {"0", "false", "no", "n"}
    has_required_sources = required_sources.issubset(available_sources)
    if str(row.get("gold_sufficient") or "").strip().lower() in {"0", "false", "no", "n"}:
        return False
    return has_required_sources


def _acceptable_action_pairs(row: dict[str, str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in _split_eval_list(row.get("gold_acceptable_actions")):
        if ":" in item:
            action_type, target_status = item.split(":", 1)
            if action_type.strip() and target_status.strip():
                pairs.add((action_type.strip(), target_status.strip()))
    gold_action = (row.get("gold_action_type") or "").strip()
    gold_status = (row.get("gold_target_status") or "").strip()
    if gold_action and gold_status:
        pairs.add((gold_action, gold_status))
    return pairs


def _action_is_acceptable(
    action: ActionProposal,
    row: dict[str, str],
    *,
    expected_sufficient: bool,
) -> bool:
    pair = (action.action_type, action.target_status)
    if pair in _acceptable_action_pairs(row):
        return True
    if not expected_sufficient and _truthy(row.get("gold_should_fallback_without_evidence")):
        return _is_conservative_action(action)
    return False


def _build_gold_triage(row: dict[str, str]) -> TriageResult:
    priority = (row.get("gold_priority") or "medium").strip()
    sla_risk = _truthy(row.get("gold_sla_risk"))
    return TriageResult(
        category=row["gold_category"],  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        urgency=_priority_to_urgency(priority),
        sla_risk=sla_risk,
        sla_risk_score=1.0 if sla_risk else 0.0,
        sla_risk_reasoning="gold rag-sensitive triage",
        sla_risk_evidence=[],
        confidence=1.0,
        reasoning="gold triage used for rag-sensitive evaluation",
        decision_source="human_edit",
    )


def _relevant_doc_ids(row: dict[str, str]) -> set[str]:
    relevant: set[str] = set()
    if _truthy(row.get("gold_requires_policy")) and row.get("gold_policy_doc_id"):
        relevant.add(f"policy:{row['gold_policy_doc_id']}")
    if _truthy(row.get("gold_requires_history")) and row.get("gold_history_doc_id"):
        relevant.add(f"history:{row['gold_history_doc_id']}")
    return relevant


def _required_doc_ids(row: dict[str, str]) -> set[str]:
    required: set[str] = set()
    if _truthy(row.get("gold_requires_policy")) and row.get("gold_policy_doc_id"):
        required.add(f"policy:{row['gold_policy_doc_id']}")
    if _truthy(row.get("gold_requires_history")) and row.get("gold_history_doc_id"):
        required.add(f"history:{row['gold_history_doc_id']}")
    return required


def _claim_coverage(reply: str, row: dict[str, str]) -> float:
    claims = _split_claims(row.get("gold_key_claims"))
    if not claims:
        return 0.0
    lowered = reply.lower()
    hits = sum(1 for claim in claims if claim.lower() in lowered)
    return round(hits / len(claims), 3)


def _is_conservative_action(action: ActionProposal) -> bool:
    if action.action_type == "request_info":
        return True
    if action.target_status in {"waiting_on_customer", "pending_human"}:
        return True
    return False


def _chunk_rows(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    batch_size = max(1, batch_size)
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _cache_context_signature(runner: TicketFlowRunner, cache_version: str) -> str:
    payload = {
        "cache_version": cache_version,
        "model_backend": runner.settings.model_backend,
        "cloud_model": runner.settings.cloud_model_name,
        "minimind_model": runner.settings.minimind_model,
        "minimind_structured_model": runner.settings.minimind_structured_model,
        "minimind_reply_model": runner.settings.minimind_reply_model,
        "rag_embed_backend": runner.settings.rag_embed_backend,
        "rag_embed_model": runner.settings.rag_embed_model,
        "rag_embed_device": runner.settings.rag_embed_device,
        "retriever_backend_used": runner.retriever.embedding_backend_name,
        "retriever_collection_name": runner.retriever.collection_name,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _sample_cache_path(
    *,
    cache_root: Path,
    context_signature: str,
    mode: str,
    row: dict[str, str],
) -> Path:
    sample_id = row.get("sample_id") or row["ticket_id"]
    safe_sample_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)
    return cache_root / context_signature / mode / f"{safe_sample_id}.json"


def _retrieve_docs(
    runner: TicketFlowRunner,
    ticket: TicketRecord,
    mode: str,
) -> tuple[list[RetrievedDoc], list[RetrievedDoc], list[RetrievedDoc]]:
    if mode == "no_rag":
        rag_docs: list[RetrievedDoc] = []
    elif mode == "kb_only":
        rag_docs = runner.retriever.baseline_retrieve(ticket).docs
    elif mode == "hybrid":
        rag_docs = runner.retriever.retrieve(ticket).docs
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unknown mode: {mode}")

    customer = runner.repository.get_customer_profile(ticket.customer_id)
    order = runner.repository.get_order_status(ticket.linked_order_id)
    context_docs = runner.knowledge_agent.retrieve(customer, order, rag_docs)
    policy_hits = [doc for doc in rag_docs if doc.source_type == "policy"]
    return rag_docs, context_docs, policy_hits


def _evaluate_row(
    *,
    runner: TicketFlowRunner,
    row: dict[str, str],
    mode: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    ticket = runner.get_ticket(row["ticket_id"])
    triage = _build_gold_triage(row)
    rag_docs, context_docs, policy_hits = _retrieve_docs(runner, ticket, mode)
    customer = runner.repository.get_customer_profile(ticket.customer_id)
    order = runner.repository.get_order_status(ticket.linked_order_id)
    gold_route_family = infer_route_family(ticket, triage)
    sufficiency_result = runner.resolution_agent.assess_sufficiency(
        ticket=ticket,
        triage=triage,
        customer=customer,
        order=order,
        policy_hits=policy_hits,
        retrieved_docs=context_docs,
    )
    action_route = build_action_route(sufficiency_result)

    proposed_action = runner.resolution_agent.propose_action(
        ticket=ticket,
        triage=triage,
        customer=customer,
        order=order,
        policy_hits=policy_hits,
        retrieved_docs=context_docs,
        sufficiency_result=sufficiency_result,
        action_route=action_route,
    )
    execution_view = SimpleNamespace(
        status="needs_handoff" if proposed_action.requires_approval else "executed",
        tool_output={},
    )
    reply_draft = runner.resolution_agent.draft_reply(
        ticket=ticket,
        triage=triage,
        proposed_action=proposed_action,
        execution_result=execution_view,
        retrieved_docs=context_docs,
        templates=runner.repository.get_reply_templates(triage.category, limit=3),
        review_decision=None,
    )
    draft_fact_check = _revalidate_final_reply_against_contract(
        ticket,
        triage,
        proposed_action,
        execution_view,
        context_docs,
        reply_draft,
        threshold=runner.resolution_agent.reply_supported_claim_threshold,
    )
    reply, fact_check = runner.resolution_agent.fact_check_reply(
        ticket=ticket,
        triage=triage,
        proposed_action=proposed_action,
        execution_result=execution_view,
        retrieved_docs=context_docs,
        response=reply_draft,
    )

    retrieved_doc_ids = {f"{doc.source_type}:{doc.doc_id}" for doc in rag_docs}
    context_doc_ids = {f"{doc.source_type}:{doc.doc_id}" for doc in context_docs}
    relevant_doc_ids = _relevant_doc_ids(row)
    required_doc_ids = _required_doc_ids(row)
    has_required_doc_evidence = (not required_doc_ids) or required_doc_ids.issubset(retrieved_doc_ids)
    required_source_types = _required_source_types(row)
    available_source_types = _available_source_types(context_docs)
    expected_sufficient = _expected_sufficient(row, available_source_types)
    min_history_strength = float(row.get("gold_min_history_strength") or 0.0)
    if (
        expected_sufficient
        and "history" in required_source_types
        and min_history_strength > 0.0
        and sufficiency_result.support_strength < min_history_strength
    ):
        expected_sufficient = False
    is_high_risk_route = gold_route_family in {"refund_candidate", "escalation_candidate"}
    action_correct = _action_is_acceptable(proposed_action, row, expected_sufficient=expected_sufficient)
    claim_coverage = _claim_coverage(reply.customer_reply, row)
    grounded_claim_coverage = claim_coverage if expected_sufficient else 0.0
    missing_required_evidence = bool(required_source_types and not expected_sufficient)

    policy_required = int(_truthy(row.get("gold_requires_policy")))
    policy_hit = int(
        bool(row.get("gold_policy_doc_id"))
        and f"policy:{row['gold_policy_doc_id']}" in retrieved_doc_ids
    )
    history_required = int(_truthy(row.get("gold_requires_history")))
    history_hit = int(
        bool(row.get("gold_history_doc_id"))
        and f"history:{row['gold_history_doc_id']}" in retrieved_doc_ids
    )

    precision_contribution = 0.0
    precision_samples = 0
    if relevant_doc_ids:
        top_k = [f"{doc.source_type}:{doc.doc_id}" for doc in rag_docs[:3]]
        matched = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
        precision_contribution = matched / 3.0
        precision_samples = 1

    context_recall_contribution = 0.0
    context_recall_samples = 0
    if required_source_types:
        matched_required = len(required_source_types & available_source_types)
        context_recall_contribution = matched_required / len(required_source_types)
        context_recall_samples = 1

    fallback_candidate = int(
        _truthy(row.get("gold_should_fallback_without_evidence")) and missing_required_evidence
    )
    evidence_action_sample = int(bool(required_source_types) and expected_sufficient and sufficiency_result.sufficient)

    tool_policy = get_tool_approval_policy(proposed_action.suggested_tool)
    tool_approval_sample = int(tool_policy.sensitive)
    tool_approval_hit = int(
        tool_policy.sensitive
        and tool_policy.approval_mode != "auto"
        and proposed_action.requires_approval
    )

    bad_case: dict[str, Any] | None = None
    if (relevant_doc_ids and not relevant_doc_ids.issubset(retrieved_doc_ids)) or not action_correct:
        bad_case = {
            "ticket_id": ticket.ticket_id,
            "mode": mode,
            "required_sources": sorted(required_source_types),
            "available_sources": sorted(available_source_types),
            "expected_sufficient": expected_sufficient,
            "required_docs": sorted(required_doc_ids),
            "retrieved_docs": sorted(retrieved_doc_ids),
            "context_docs": sorted(context_doc_ids),
            "predicted_action": proposed_action.action_type,
            "predicted_target_status": proposed_action.target_status,
            "gold_action": row["gold_action_type"],
            "gold_target_status": row["gold_target_status"],
            "route_family": sufficiency_result.route_family,
            "gold_route_family": gold_route_family,
            "sufficient": sufficiency_result.sufficient,
            "missing_sources": sufficiency_result.missing_sources,
            "reply_claim_coverage": grounded_claim_coverage,
            "reply_fact_check_passed": fact_check.passed,
        }

    return {
        "sample_id": row.get("sample_id") or row["ticket_id"],
        "ticket_id": ticket.ticket_id,
        "mode": mode,
        "seconds": round(time.perf_counter() - started_at, 3),
        "route_family_hit": int(sufficiency_result.route_family == gold_route_family),
        "sufficiency_hit": int(sufficiency_result.sufficient == expected_sufficient),
        "high_risk_sufficiency_candidate": int(is_high_risk_route and expected_sufficient),
        "high_risk_sufficiency_hit": int(is_high_risk_route and expected_sufficient and sufficiency_result.sufficient),
        "policy_required": policy_required,
        "policy_hit": policy_hit,
        "history_required": history_required,
        "history_hit": history_hit,
        "precision_samples": precision_samples,
        "precision_contribution": round(precision_contribution, 6),
        "context_recall_samples": context_recall_samples,
        "context_recall_contribution": round(context_recall_contribution, 6),
        "evidence_sample": evidence_action_sample,
        "evidence_action_hit": int(evidence_action_sample and action_correct),
        "reply_usage_sample": 1,
        "reply_usage_hit": int(grounded_claim_coverage > 0.0),
        "reply_claim_coverage": grounded_claim_coverage,
        "reply_draft_fact_check_sample": 1,
        "reply_draft_fact_check_hit": int(bool(draft_fact_check.get("review_passed"))),
        "reply_draft_groundedness": float(draft_fact_check.get("supported_claim_ratio") or 0.0),
        "reply_fact_check_sample": 1,
        "reply_fact_check_hit": int(fact_check.passed),
        "reply_groundedness": float(reply.review_supported_claim_ratio or 0.0),
        "fallback_candidate": fallback_candidate,
        "fallback_hit": int(fallback_candidate and _is_conservative_action(proposed_action)),
        "tool_approval_sample": tool_approval_sample,
        "tool_approval_hit": tool_approval_hit,
        "bad_case": bad_case,
    }


def _aggregate_mode_results(
    *,
    mode: str,
    rows: list[dict[str, str]],
    sample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    policy_required = sum(int(result["policy_required"]) for result in sample_results)
    policy_hit = sum(int(result["policy_hit"]) for result in sample_results)
    history_required = sum(int(result["history_required"]) for result in sample_results)
    history_hit = sum(int(result["history_hit"]) for result in sample_results)
    precision_samples = sum(int(result["precision_samples"]) for result in sample_results)
    precision_sum = sum(float(result["precision_contribution"]) for result in sample_results)
    context_recall_samples = sum(int(result["context_recall_samples"]) for result in sample_results)
    context_recall_sum = sum(float(result["context_recall_contribution"]) for result in sample_results)
    evidence_samples = sum(int(result["evidence_sample"]) for result in sample_results)
    evidence_action_correct = sum(int(result["evidence_action_hit"]) for result in sample_results)
    reply_usage_samples = sum(int(result["reply_usage_sample"]) for result in sample_results)
    reply_usage_hits = sum(int(result["reply_usage_hit"]) for result in sample_results)
    reply_claim_coverage_sum = sum(float(result["reply_claim_coverage"]) for result in sample_results)
    reply_draft_fact_check_samples = sum(int(result["reply_draft_fact_check_sample"]) for result in sample_results)
    reply_draft_fact_check_hits = sum(int(result["reply_draft_fact_check_hit"]) for result in sample_results)
    reply_draft_groundedness_sum = sum(float(result["reply_draft_groundedness"]) for result in sample_results)
    fallback_candidates = sum(int(result["fallback_candidate"]) for result in sample_results)
    fallback_hits = sum(int(result["fallback_hit"]) for result in sample_results)
    route_samples = len(sample_results)
    route_hits = sum(int(result["route_family_hit"]) for result in sample_results)
    sufficiency_samples = len(sample_results)
    sufficiency_hits = sum(int(result["sufficiency_hit"]) for result in sample_results)
    high_risk_sufficiency_candidates = sum(int(result["high_risk_sufficiency_candidate"]) for result in sample_results)
    high_risk_sufficiency_hits = sum(int(result["high_risk_sufficiency_hit"]) for result in sample_results)
    tool_approval_samples = sum(int(result["tool_approval_sample"]) for result in sample_results)
    tool_approval_hits = sum(int(result["tool_approval_hit"]) for result in sample_results)
    reply_fact_check_samples = sum(int(result["reply_fact_check_sample"]) for result in sample_results)
    reply_fact_check_hits = sum(int(result["reply_fact_check_hit"]) for result in sample_results)
    reply_groundedness_sum = sum(float(result["reply_groundedness"]) for result in sample_results)
    bad_cases = [result["bad_case"] for result in sample_results if result.get("bad_case")][:12]

    return {
        "mode": mode,
        "sample_size": len(rows),
        "cache_hits": sum(int(bool(result.get("_cache_hit"))) for result in sample_results),
        "cache_misses": sum(int(not bool(result.get("_cache_hit"))) for result in sample_results),
        "total_seconds": round(sum(float(result["seconds"]) for result in sample_results), 3),
        "route_family_accuracy": round(route_hits / max(route_samples, 1), 3),
        "sufficiency_accuracy": round(sufficiency_hits / max(sufficiency_samples, 1), 3),
        "sufficiency_recall_for_high_risk": round(high_risk_sufficiency_hits / max(high_risk_sufficiency_candidates, 1), 3),
        "policy_evidence_recall": round(policy_hit / max(policy_required, 1), 3),
        "history_evidence_recall": round(history_hit / max(history_required, 1), 3),
        "context_precision_at_3": round(precision_sum / max(precision_samples, 1), 3),
        "context_recall": round(context_recall_sum / max(context_recall_samples, 1), 3),
        "evidence_based_action_accuracy": round(evidence_action_correct / max(evidence_samples, 1), 3),
        "tool_approval_interception_accuracy": round(tool_approval_hits / max(tool_approval_samples, 1), 3),
        "reply_evidence_usage_rate": round(reply_usage_hits / max(reply_usage_samples, 1), 3),
        "reply_draft_fact_check_pass_rate": round(reply_draft_fact_check_hits / max(reply_draft_fact_check_samples, 1), 3),
        "reply_groundedness_before_rewrite": round(reply_draft_groundedness_sum / max(reply_draft_fact_check_samples, 1), 3),
        "reply_fact_check_pass_rate": round(reply_fact_check_hits / max(reply_fact_check_samples, 1), 3),
        "reply_groundedness_after_rewrite": round(reply_groundedness_sum / max(reply_fact_check_samples, 1), 3),
        "reply_claim_coverage_mean": round(reply_claim_coverage_sum / max(reply_usage_samples, 1), 3),
        "conservative_fallback_rate_without_evidence": round(fallback_hits / max(fallback_candidates, 1), 3),
        "bad_cases": bad_cases,
    }


def _evaluate_mode(
    runner: TicketFlowRunner,
    rows: list[dict[str, str]],
    mode: str,
    *,
    cache_root: Path,
    context_signature: str,
    batch_size: int,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []
    mode_partial_path = cache_root / context_signature / mode / "_mode_partial.json"
    total_rows = len(rows)
    batches = _chunk_rows(rows, batch_size)
    completed = 0

    for batch_index, batch_rows in enumerate(batches, start=1):
        batch_results: list[dict[str, Any]] = []
        for row in batch_rows:
            cache_path = _sample_cache_path(
                cache_root=cache_root,
                context_signature=context_signature,
                mode=mode,
                row=row,
            )
            if cache_path.exists():
                sample_result = json.loads(cache_path.read_text(encoding="utf-8"))
                sample_result["_cache_hit"] = True
            else:
                sample_result = _evaluate_row(runner=runner, row=row, mode=mode)
                sample_result["_cache_hit"] = False
                _write_json_atomic(cache_path, sample_result)
            batch_results.append(sample_result)
        sample_results.extend(batch_results)
        completed += len(batch_rows)
        partial = _aggregate_mode_results(mode=mode, rows=rows[:completed], sample_results=sample_results)
        partial["completed_samples"] = completed
        partial["total_samples"] = total_rows
        partial["batch_index"] = batch_index
        partial["batch_count"] = len(batches)
        _write_json_atomic(mode_partial_path, partial)
        print(
            f"[rag-sensitive] mode={mode} batch={batch_index}/{len(batches)} "
            f"completed={completed}/{total_rows} cache_hits={partial['cache_hits']} cache_misses={partial['cache_misses']}"
        )

    final_report = _aggregate_mode_results(mode=mode, rows=rows, sample_results=sample_results)
    _write_json_atomic(mode_partial_path, final_report)
    return final_report


def evaluate_rag_sensitive(
    *,
    project_root: Path,
    eval_csv: Path,
    overrides: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    cache_version: str = DEFAULT_CACHE_VERSION,
    batch_size: int = 5,
    clear_cache: bool = False,
) -> dict[str, Any]:
    rows = _load_rows(eval_csv)
    runner = TicketFlowRunner.from_project_root(project_root, overrides=overrides)
    cache_root = cache_dir or (project_root / "reports" / "cache" / "rag_sensitive")
    context_signature = _cache_context_signature(runner, cache_version)
    if clear_cache:
        shutil.rmtree(cache_root / context_signature, ignore_errors=True)

    reports = {
        mode: _evaluate_mode(
            runner,
            rows,
            mode,
            cache_root=cache_root,
            context_signature=context_signature,
            batch_size=batch_size,
        )
        for mode in ("no_rag", "kb_only", "hybrid")
    }

    def _delta(metric: str, new_mode: str, base_mode: str) -> dict[str, float]:
        new_value = reports[new_mode][metric]
        base_value = reports[base_mode][metric]
        return {
            base_mode: base_value,
            new_mode: new_value,
            "delta": round(new_value - base_value, 3),
        }

    metrics = [
        "route_family_accuracy",
        "sufficiency_accuracy",
        "sufficiency_recall_for_high_risk",
        "policy_evidence_recall",
        "history_evidence_recall",
        "context_precision_at_3",
        "context_recall",
        "evidence_based_action_accuracy",
        "tool_approval_interception_accuracy",
        "reply_evidence_usage_rate",
        "reply_draft_fact_check_pass_rate",
        "reply_groundedness_before_rewrite",
        "reply_fact_check_pass_rate",
        "reply_groundedness_after_rewrite",
        "reply_claim_coverage_mean",
        "conservative_fallback_rate_without_evidence",
    ]
    return {
        "project": "TicketFlow RAG-sensitive evaluation",
        "eval_csv": str(eval_csv),
        "sample_size": len(rows),
        "cache_version": cache_version,
        "cache_dir": str(cache_root),
        "cache_context_signature": context_signature,
        "batch_size": batch_size,
        "reports": reports,
        "deltas": {
            "kb_only_vs_no_rag": {metric: _delta(metric, "kb_only", "no_rag") for metric in metrics},
            "hybrid_vs_kb_only": {metric: _delta(metric, "hybrid", "kb_only") for metric in metrics},
            "hybrid_vs_no_rag": {metric: _delta(metric, "hybrid", "no_rag") for metric in metrics},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RAG-sensitive benchmark for TicketFlow.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "gold_eval" / "rag_sensitive_eval_v1.csv",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "rag_sensitive_eval_report_v1.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "cache" / "rag_sensitive",
    )
    parser.add_argument("--cache-version", type=str, default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()
    report = evaluate_rag_sensitive(
        project_root=args.project_root,
        eval_csv=args.eval_csv,
        cache_dir=args.cache_dir,
        cache_version=args.cache_version,
        batch_size=args.batch_size,
        clear_cache=args.clear_cache,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
