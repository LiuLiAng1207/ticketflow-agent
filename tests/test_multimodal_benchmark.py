from __future__ import annotations

import csv
import json
from pathlib import Path

from ticketflow.build_multimodal_ticket_eval import build_multimodal_ticket_eval
from ticketflow.eval_multimodal_ticket import evaluate_multimodal_ticket_eval


def test_build_multimodal_ticket_eval_writes_balanced_csv(ticketflow_project: Path):
    output_path = ticketflow_project / "data" / "eval" / "multimodal_ticket_eval_v1.csv"

    summary = build_multimodal_ticket_eval(project_root=ticketflow_project, output_path=output_path, target_size=25)

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8-sig", newline="")))
    assert len(rows) == 25
    assert summary["sample_count"] == 25
    assert {"payment_verified", "payment_unverified", "error_screenshot", "account_screenshot", "document_form"} <= set(summary["by_case_type"])
    assert rows[0]["attachment_ocr_text"]
    assert json.loads(rows[0]["gold_entities_json"])


def test_evaluate_multimodal_ticket_eval_outputs_layered_metrics(ticketflow_project: Path):
    dataset_path = ticketflow_project / "data" / "eval" / "multimodal_ticket_eval_v1_small.csv"
    report_path = ticketflow_project / "reports" / "multimodal_ticket_eval_v1_small.json"
    build_multimodal_ticket_eval(project_root=ticketflow_project, output_path=dataset_path, target_size=8)

    report = evaluate_multimodal_ticket_eval(
        project_root=ticketflow_project,
        dataset_path=dataset_path,
        output_path=report_path,
        limit=5,
        overrides={"RAG_EMBED_BACKEND": "hash"},
    )

    assert report_path.exists()
    assert report["sample_count"] == 5
    assert "attachment_parse_success_rate" in report["metrics"]
    assert "multimodal_evidence_recall" in report["metrics"]
    assert "attachment_evidence_mrr" in report["metrics"]
    assert "attachment_entity_accuracy" in report["metrics"]
    assert "sufficiency_accuracy" in report["metrics"]
    assert report["samples"]
