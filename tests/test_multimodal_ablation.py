from __future__ import annotations

from pathlib import Path

from ticketflow.build_multimodal_ticket_eval import build_multimodal_ticket_eval
from ticketflow.eval_multimodal_ablation import evaluate_multimodal_ablation
from ticketflow.eval_multimodal_ticket import evaluate_multimodal_ticket_eval


def test_text_only_ablation_removes_attachment_evidence(ticketflow_project: Path):
    dataset_path = ticketflow_project / "data" / "eval" / "multimodal_ticket_eval_ablation_small.csv"
    enabled_report_path = ticketflow_project / "reports" / "multimodal_enabled_small.json"
    text_only_report_path = ticketflow_project / "reports" / "multimodal_text_only_small.json"
    build_multimodal_ticket_eval(project_root=ticketflow_project, output_path=dataset_path, target_size=8)

    enabled = evaluate_multimodal_ticket_eval(
        project_root=ticketflow_project,
        dataset_path=dataset_path,
        output_path=enabled_report_path,
        limit=5,
        overrides={"RAG_EMBED_BACKEND": "hash"},
    )
    text_only = evaluate_multimodal_ticket_eval(
        project_root=ticketflow_project,
        dataset_path=dataset_path,
        output_path=text_only_report_path,
        limit=5,
        overrides={"RAG_EMBED_BACKEND": "hash"},
        disable_attachments=True,
    )

    assert enabled["metrics"]["multimodal_evidence_recall"] == 1.0
    assert enabled["metrics"]["attachment_entity_accuracy"] > text_only["metrics"]["attachment_entity_accuracy"]
    assert text_only["metrics"]["multimodal_evidence_recall"] == 0.0
    assert text_only["metrics"]["attachment_parse_success_rate"] == 0.0


def test_multimodal_ablation_report_contains_lift_metrics(ticketflow_project: Path):
    dataset_path = ticketflow_project / "data" / "eval" / "multimodal_ticket_eval_ablation_report.csv"
    output_path = ticketflow_project / "reports" / "multimodal_ablation_report.json"
    build_multimodal_ticket_eval(project_root=ticketflow_project, output_path=dataset_path, target_size=10)

    report = evaluate_multimodal_ablation(
        project_root=ticketflow_project,
        dataset_path=dataset_path,
        output_path=output_path,
        limit=6,
        overrides={"RAG_EMBED_BACKEND": "hash"},
    )

    assert report["dataset"] == "multimodal_ticket_eval_v1_ablation"
    assert report["enabled"]["sample_count"] == 6
    assert report["text_only"]["sample_count"] == 6
    assert report["lift"]["multimodal_evidence_recall"] == 1.0
    assert report["lift"]["attachment_parse_success_rate"] == 1.0
    assert output_path.exists()
