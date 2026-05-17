from __future__ import annotations

from ticketflow.models import AttachmentEvidence, RetrievedDoc
from ticketflow.multimodal_eval import score_multimodal_state


def test_score_multimodal_state_reports_attachment_recall_and_mrr():
    evidence = AttachmentEvidence(
        evidence_id="attachment:ATT-1",
        attachment_id="ATT-1",
        ticket_id="TCK-1",
        evidence_type="payment_screenshot",
        extracted_text="支付成功 订单号 ORD-0001",
        confidence=0.9,
    )
    state = {
        "attachment_evidence": [evidence],
        "retrieved_docs": [
            RetrievedDoc(doc_id="KB-1", source_type="kb", title="kb", snippet="kb"),
            RetrievedDoc(doc_id="attachment:ATT-1", source_type="attachment", title="attachment", snippet="payment"),
        ],
    }

    metrics = score_multimodal_state(state, expected_evidence_types={"payment_screenshot"})

    assert metrics["attachment_parse_success"] == 1.0
    assert metrics["multimodal_evidence_recall"] == 1.0
    assert metrics["attachment_evidence_mrr"] == 0.5
