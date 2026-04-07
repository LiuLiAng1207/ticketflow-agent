from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .graph import TicketFlowRunner
from .models import InvocationResult, ReviewDecision


def _load_eval_ticket_ids(path: Path, limit: int | None = None) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("gold_split") or "").strip() == "eval"]
    ticket_ids = [row["ticket_id"] for row in rows]
    return ticket_ids[:limit] if limit is not None else ticket_ids


def _complete_ticket(runner: TicketFlowRunner, ticket_id: str) -> InvocationResult:
    result = runner.run_ticket(ticket_id)
    while result.interrupted:
        thread_id = result.state["thread_id"]
        result = runner.resume_ticket(thread_id, ReviewDecision(decision="approve"))
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / max(denominator, 1), 3)


def _evaluate_backend(project_root: Path, overrides: dict[str, str], ticket_ids: list[str], label: str) -> dict[str, Any]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides=overrides)
    triage_local = 0
    action_local = 0
    reply_local = 0
    reply_usable = 0
    reply_grounded = 0
    policy_violations = 0
    hallucinations = 0
    triage_fallbacks = 0
    action_fallbacks = 0
    reply_fallbacks = 0
    backend_errors = 0
    bad_cases: list[dict[str, Any]] = []

    for ticket_id in ticket_ids:
        try:
            result = _complete_ticket(runner, ticket_id)
        except Exception as exc:  # pragma: no cover - runtime robustness for flaky local backends
            backend_errors += 1
            bad_cases.append(
                {
                    "ticket_id": ticket_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        state = result.state
        triage = state["triage_result"]
        action = state["proposed_action"]
        reply = state["draft_reply"]
        execution = state["execution_result"]

        triage_local += int(getattr(triage, "decision_source", "") == "llm_minimind")
        action_local += int(getattr(action, "decision_source", "") == "llm_minimind")
        reply_local += int(getattr(reply, "draft_source", "") == "llm_minimind")
        triage_fallbacks += int(bool(getattr(triage, "fallback_reason", None)))
        action_fallbacks += int(bool(getattr(action, "fallback_reason", None)))
        reply_fallbacks += int(bool(getattr(reply, "fallback_reason", None)))

        usable = bool(getattr(reply, "review_passed", False)) and bool(getattr(reply, "customer_reply", "").strip())
        grounded = bool(getattr(reply, "review_grounded", False))
        policy_safe = bool(getattr(reply, "review_policy_safe", True))
        hallucination = bool(getattr(reply, "review_hallucination_risk", False))

        reply_usable += int(usable)
        reply_grounded += int(grounded)
        policy_violations += int(not policy_safe)
        hallucinations += int(hallucination)

        if not usable or not grounded or not policy_safe or hallucination:
            bad_cases.append(
                {
                    "ticket_id": ticket_id,
                    "triage_source": getattr(triage, "decision_source", None),
                    "action_source": getattr(action, "decision_source", None),
                    "reply_source": getattr(reply, "draft_source", None),
                    "reply_review_passed": getattr(reply, "review_passed", None),
                    "reply_grounded": getattr(reply, "review_grounded", None),
                    "reply_policy_safe": getattr(reply, "review_policy_safe", None),
                    "reply_hallucination_risk": getattr(reply, "review_hallucination_risk", None),
                    "execution_status": getattr(execution, "status", None),
                    "reply_fallback_reason": getattr(reply, "fallback_reason", None),
                }
            )

    sample_size = len(ticket_ids)
    structured_total = sample_size * 2
    return {
        "backend": label,
        "sample_size": sample_size,
        "metrics": {
            "local_structured_node_coverage_rate": _safe_rate(triage_local + action_local, structured_total),
            "local_triage_node_coverage_rate": _safe_rate(triage_local, sample_size),
            "local_action_node_coverage_rate": _safe_rate(action_local, sample_size),
            "local_reply_node_coverage_rate": _safe_rate(reply_local, sample_size),
            "reply_usable_rate": _safe_rate(reply_usable, sample_size),
            "reply_grounded_rate": _safe_rate(reply_grounded, sample_size),
            "policy_violation_rate": _safe_rate(policy_violations, sample_size),
            "hallucination_rate": _safe_rate(hallucinations, sample_size),
            "triage_fallback_rate": _safe_rate(triage_fallbacks, sample_size),
            "action_fallback_rate": _safe_rate(action_fallbacks, sample_size),
            "reply_fallback_rate": _safe_rate(reply_fallbacks, sample_size),
            "backend_error_rate": _safe_rate(backend_errors, sample_size),
        },
        "bad_cases": bad_cases[:20],
    }


def _delta(new_value: float, old_value: float) -> dict[str, float]:
    return {
        "baseline": round(old_value, 3),
        "new": round(new_value, 3),
        "absolute_delta": round(new_value - old_value, 3),
    }


def _evaluate_backend_safe(
    project_root: Path,
    overrides: dict[str, str],
    ticket_ids: list[str],
    label: str,
) -> dict[str, Any]:
    try:
        report = _evaluate_backend(project_root, overrides, ticket_ids, label)
        report["status"] = "ok"
        return report
    except Exception as exc:  # pragma: no cover - runtime guard for flaky model backends
        return {
            "backend": label,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "sample_size": len(ticket_ids),
            "metrics": {},
            "bad_cases": [],
        }


def evaluate_split_report(
    *,
    project_root: Path,
    gold_csv: Path,
    limit: int | None,
    base_base_url: str,
    base_model: str,
    base_api_key: str,
    mixed_base_url: str,
    mixed_model: str,
    mixed_api_key: str,
    structured_base_url: str,
    structured_model: str,
    structured_api_key: str,
    reply_base_url: str,
    reply_model: str,
    reply_api_key: str,
) -> dict[str, Any]:
    ticket_ids = _load_eval_ticket_ids(gold_csv, limit=limit)

    base_report = _evaluate_backend_safe(
        project_root,
        overrides={
            "MODEL_BACKEND": "cloud_api",
        },
        ticket_ids=ticket_ids,
        label="cloud_mainchain",
    )
    base_local_report = _evaluate_backend_safe(
        project_root,
        overrides={
            "MODEL_BACKEND": "minimind_api",
            "OPENAI_COMPAT_BASE_URL": "",
            "OPENAI_COMPAT_MODEL": "",
            "OPENAI_COMPAT_API_KEY": "",
            "MINIMIND_BASE_URL": base_base_url,
            "MINIMIND_MODEL": base_model,
            "MINIMIND_API_KEY": base_api_key,
            "MINIMIND_ENABLE_TRIAGE": "true",
            "MINIMIND_ENABLE_DRAFT_REPLY": "true",
        },
        ticket_ids=ticket_ids,
        label="base_local_model",
    )
    mixed_report = _evaluate_backend_safe(
        project_root,
        overrides={
            "MODEL_BACKEND": "minimind_api",
            "OPENAI_COMPAT_BASE_URL": "",
            "OPENAI_COMPAT_MODEL": "",
            "OPENAI_COMPAT_API_KEY": "",
            "MINIMIND_BASE_URL": mixed_base_url,
            "MINIMIND_MODEL": mixed_model,
            "MINIMIND_API_KEY": mixed_api_key,
            "MINIMIND_ENABLE_TRIAGE": "true",
            "MINIMIND_ENABLE_DRAFT_REPLY": "true",
        },
        ticket_ids=ticket_ids,
        label="old_single_lora_mixed",
    )
    split_report = _evaluate_backend_safe(
        project_root,
        overrides={
            "MODEL_BACKEND": "minimind_split",
            "OPENAI_COMPAT_BASE_URL": "",
            "OPENAI_COMPAT_MODEL": "",
            "OPENAI_COMPAT_API_KEY": "",
            "MINIMIND_STRUCTURED_BASE_URL": structured_base_url,
            "MINIMIND_STRUCTURED_MODEL": structured_model,
            "MINIMIND_STRUCTURED_API_KEY": structured_api_key,
            "MINIMIND_REPLY_BASE_URL": reply_base_url,
            "MINIMIND_REPLY_MODEL": reply_model,
            "MINIMIND_REPLY_API_KEY": reply_api_key,
        },
        ticket_ids=ticket_ids,
        label="new_split_lora",
    )

    metric_keys = list(
        set(base_report["metrics"].keys())
        | set(base_local_report["metrics"].keys())
        | set(mixed_report["metrics"].keys())
        | set(split_report["metrics"].keys())
    )

    def _delta_against(reference: dict[str, Any]) -> dict[str, dict[str, float] | dict[str, Any]]:
        if reference.get("status") != "ok":
            return {
                "status": "unavailable",
                "reason": reference.get("error", "unknown backend failure"),
            }
        return {
            key: _delta(split_report["metrics"][key], reference["metrics"][key])
            for key in metric_keys
            if key in split_report["metrics"] and key in reference["metrics"]
        }

    return {
        "project": "TicketFlow MiniMind Split Integration",
        "gold_csv": str(gold_csv),
        "sample_size": len(ticket_ids),
        "reports": {
            "cloud_mainchain": base_report,
            "base_local_model": base_local_report,
            "old_single_lora_mixed": mixed_report,
            "new_split_lora": split_report,
        },
        "deltas": {
            "split_vs_cloud": _delta_against(base_report),
            "split_vs_base_local": _delta_against(base_local_report),
            "split_vs_old_mixed": _delta_against(mixed_report),
        },
        "metric_labels_cn": {
            "local_structured_node_coverage_rate": "本地结构化节点覆盖率",
            "local_triage_node_coverage_rate": "本地分诊节点覆盖率",
            "local_action_node_coverage_rate": "本地动作决策节点覆盖率",
            "local_reply_node_coverage_rate": "本地回复节点覆盖率",
            "reply_usable_rate": "回复可用率",
            "reply_grounded_rate": "回复有依据率",
            "policy_violation_rate": "策略违规率",
            "hallucination_rate": "明显事实编造率",
            "triage_fallback_rate": "分诊回退率",
            "action_fallback_rate": "动作节点回退率",
            "reply_fallback_rate": "回复节点回退率",
            "backend_error_rate": "后端错误率",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TicketFlow with old single-LoRA vs new split-LoRA integration.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gold-csv", type=Path, default=Path(r"D:\Study\shixi\public_structured_gold_canonical_v3.csv"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--base-base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-api-key", default="sk-local")
    parser.add_argument("--mixed-base-url", required=True)
    parser.add_argument("--mixed-model", required=True)
    parser.add_argument("--mixed-api-key", default="sk-local")
    parser.add_argument("--structured-base-url", required=True)
    parser.add_argument("--structured-model", required=True)
    parser.add_argument("--structured-api-key", default="sk-local")
    parser.add_argument("--reply-base-url", required=True)
    parser.add_argument("--reply-model", required=True)
    parser.add_argument("--reply-api-key", default="sk-local")
    parser.add_argument("--report-path", type=Path, default=Path(r"D:\Study\shixi\split_mainchain_report_v1.json"))
    args = parser.parse_args()

    report = evaluate_split_report(
        project_root=args.project_root,
        gold_csv=args.gold_csv,
        limit=args.limit,
        base_base_url=args.base_base_url,
        base_model=args.base_model,
        base_api_key=args.base_api_key,
        mixed_base_url=args.mixed_base_url,
        mixed_model=args.mixed_model,
        mixed_api_key=args.mixed_api_key,
        structured_base_url=args.structured_base_url,
        structured_model=args.structured_model,
        structured_api_key=args.structured_api_key,
        reply_base_url=args.reply_base_url,
        reply_model=args.reply_model,
        reply_api_key=args.reply_api_key,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
