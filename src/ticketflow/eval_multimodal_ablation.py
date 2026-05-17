from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_multimodal_ticket import _parse_override, evaluate_multimodal_ticket_eval


def _metric_lift(enabled: dict[str, float], text_only: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(enabled) | set(text_only))
    return {key: round(float(enabled.get(key, 0.0)) - float(text_only.get(key, 0.0)), 3) for key in keys}


def evaluate_multimodal_ablation(
    *,
    project_root: Path,
    dataset_path: Path,
    output_path: Path,
    limit: int | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    enabled_output = output_path.with_name(output_path.stem + "_with_attachment.json")
    text_only_output = output_path.with_name(output_path.stem + "_text_only.json")

    enabled = evaluate_multimodal_ticket_eval(
        project_root=project_root,
        dataset_path=dataset_path,
        output_path=enabled_output,
        limit=limit,
        overrides=overrides,
        disable_attachments=False,
        thread_prefix="mm-enabled",
    )
    text_only = evaluate_multimodal_ticket_eval(
        project_root=project_root,
        dataset_path=dataset_path,
        output_path=text_only_output,
        limit=limit,
        overrides=overrides,
        disable_attachments=True,
        thread_prefix="mm-text-only",
    )

    report: dict[str, object] = {
        "dataset": "multimodal_ticket_eval_v1_ablation",
        "dataset_path": str(dataset_path),
        "sample_count": enabled["sample_count"],
        "enabled": {
            "report_path": str(enabled_output),
            "sample_count": enabled["sample_count"],
            "metrics": enabled["metrics"],
        },
        "text_only": {
            "report_path": str(text_only_output),
            "sample_count": text_only["sample_count"],
            "metrics": text_only["metrics"],
        },
        "lift": _metric_lift(enabled["metrics"], text_only["metrics"]),
        "notes": (
            "This ablation uses the same tickets twice. The text-only run withholds all benchmark "
            "attachments, so metric lift measures the contribution of attachment evidence rather than "
            "a changed ticket distribution."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multimodal vs text-only TicketFlow ablation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    dataset = args.dataset or args.project_root / "data" / "eval" / "multimodal_ticket_eval_v1.csv"
    output = args.output or args.project_root / "reports" / "multimodal_ablation_eval_v1.json"
    report = evaluate_multimodal_ablation(
        project_root=args.project_root,
        dataset_path=dataset,
        output_path=output,
        limit=args.limit,
        overrides=_parse_override(args.override),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sample_count": report["sample_count"],
                "lift": report["lift"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
