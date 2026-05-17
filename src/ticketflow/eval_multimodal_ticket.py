from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .graph import TicketFlowRunner
from .models import ActionProposal
from .multimodal_eval import score_multimodal_state


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _safe_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def _entity_accuracy(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    hits = 0
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float) or isinstance(actual_value, float):
            try:
                if abs(float(actual_value) - float(expected_value)) < 1e-6:
                    hits += 1
                continue
            except Exception:
                pass
        if str(actual_value).strip().lower() == str(expected_value).strip().lower():
            hits += 1
    return hits / max(len(expected), 1)


def _attachment_entities(state: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in state.get("attachment_evidence", []) or []:
        entities = item.entities if hasattr(item, "entities") else item.get("entities", {})
        merged.update(entities or {})
    return merged


def _action_from_state(state: dict[str, Any]) -> ActionProposal | None:
    action = state.get("proposed_action")
    if action is None:
        return None
    if isinstance(action, ActionProposal):
        return action
    return ActionProposal.model_validate(action)


def _is_conservative(action: ActionProposal | None) -> bool:
    if action is None:
        return False
    return action.action_type == "request_info" or action.target_status in {"waiting_on_customer", "pending_human"}


def _mean(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 3)


def evaluate_multimodal_ticket_eval(
    *,
    project_root: Path,
    dataset_path: Path,
    output_path: Path,
    limit: int | None = None,
    overrides: dict[str, str] | None = None,
    disable_attachments: bool = False,
    thread_prefix: str = "mm-eval",
) -> dict[str, Any]:
    rows = _load_rows(dataset_path)
    if limit is not None:
        rows = rows[:limit]

    runner = TicketFlowRunner.from_project_root(project_root, overrides=overrides)
    try:
        runner.reset_demo_data()
        with runner.repository.connect() as conn:
            conn.execute("DELETE FROM attachment_evidence")
            conn.execute("DELETE FROM ticket_attachments")
            conn.commit()

        metric_lists: dict[str, list[float]] = defaultdict(list)
        samples: list[dict[str, Any]] = []
        by_case_type: Counter[str] = Counter()

        for row in rows:
            by_case_type[row["case_type"]] += 1
            if not disable_attachments:
                runner.repository.create_ticket_attachment(
                    ticket_id=row["ticket_id"],
                    filename=row["attachment_filename"],
                    file_type=row["attachment_file_type"],
                    source_dataset=row["attachment_source_dataset"],
                    ocr_text=row["attachment_ocr_text"],
                    visual_summary=row["attachment_visual_summary"],
                    metadata={"benchmark_sample_id": row["sample_id"], "benchmark_case_type": row["case_type"]},
                )
            result = runner.run_ticket(row["ticket_id"], thread_id=f"{thread_prefix}-{row['sample_id']}")
            state = result.state
            mm_metrics = score_multimodal_state(state, expected_evidence_types={row["gold_evidence_type"]})
            for key, value in mm_metrics.items():
                metric_lists[key].append(float(value))

            entity_accuracy = _entity_accuracy(_attachment_entities(state), _safe_json(row["gold_entities_json"]))
            metric_lists["attachment_entity_accuracy"].append(entity_accuracy)

            sufficiency = state.get("sufficiency_result")
            expected_sufficient = _truthy(row["gold_sufficient"])
            actual_sufficient = bool(getattr(sufficiency, "sufficient", False)) if sufficiency is not None else False
            metric_lists["sufficiency_accuracy"].append(float(actual_sufficient == expected_sufficient))

            action = _action_from_state(state)
            expected_pair = (row["gold_action_type"], row["gold_target_status"])
            actual_pair = (action.action_type, action.target_status) if action is not None else ("", "")
            action_match = actual_pair == expected_pair
            if not expected_sufficient and _truthy(row["gold_should_fallback_without_evidence"]):
                action_match = _is_conservative(action)
            metric_lists["evidence_based_action_accuracy"].append(float(action_match))

            expected_approval = _truthy(row["gold_requires_approval"])
            metric_lists["tool_approval_interception_accuracy"].append(float(bool(result.interrupted) == expected_approval))

            if _truthy(row["gold_should_fallback_without_evidence"]):
                metric_lists["conservative_fallback_rate_without_evidence"].append(float(_is_conservative(action)))

            fact_check = state.get("reply_fact_check")
            if fact_check is not None:
                metric_lists["reply_fact_check_pass_rate"].append(float(bool(getattr(fact_check, "passed", False))))

            samples.append(
                {
                    "sample_id": row["sample_id"],
                    "ticket_id": row["ticket_id"],
                    "case_type": row["case_type"],
                    "expected_evidence_type": row["gold_evidence_type"],
                    "expected_entities": _safe_json(row["gold_entities_json"]),
                    "actual_entities": _attachment_entities(state),
                    "actual_action": actual_pair,
                    "expected_action": expected_pair,
                    "interrupted_for_approval": result.interrupted,
                    "sufficiency": actual_sufficient,
                    "multimodal_metrics": mm_metrics,
                    "entity_accuracy": round(entity_accuracy, 3),
                }
            )

        metrics = {key: _mean(values) for key, values in sorted(metric_lists.items())}
        if "attachment_parse_success" in metrics:
            metrics["attachment_parse_success_rate"] = metrics["attachment_parse_success"]
        report = {
            "dataset": "multimodal_ticket_eval_v1",
            "dataset_path": str(dataset_path),
            "mode": "text_only_no_attachment" if disable_attachments else "multimodal_with_attachment",
            "sample_count": len(rows),
            "by_case_type": dict(by_case_type),
            "metrics": metrics,
            "samples": samples,
            "notes": "Attachment evidence is auxiliary; high-risk actions still require core policy/order/history evidence.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        runner.close()


def _parse_override(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TicketFlow on the multimodal ticket benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--disable-attachments",
        action="store_true",
        help="Run a text-only ablation by withholding all benchmark attachments.",
    )
    args = parser.parse_args()

    dataset = args.dataset or args.project_root / "data" / "eval" / "multimodal_ticket_eval_v1.csv"
    output = args.output or args.project_root / "reports" / "multimodal_ticket_eval_v1.json"
    report = evaluate_multimodal_ticket_eval(
        project_root=args.project_root,
        dataset_path=dataset,
        output_path=output,
        limit=args.limit,
        overrides=_parse_override(args.override),
        disable_attachments=args.disable_attachments,
    )
    print(json.dumps({"output": str(output), "sample_count": report["sample_count"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
