from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AttachmentEvidence, AttachmentRecord, Citation, RetrievedDoc


_ORDER_RE = re.compile(r"\b(?:ORD|ORDER)-?\d{3,}\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(?:金额|合计|总计|支付|total|amount)[^\d]{0,8}([0-9]+(?:\.[0-9]{1,2})?)", re.IGNORECASE)
_ERROR_RE = re.compile(r"\b(?:error\s*)?(?:40[134]|50[0234]|[A-Z]{2,5}-\d{2,6})\b", re.IGNORECASE)


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    if match.lastindex:
        return match.group(1)
    return match.group(0)


def _detect_evidence_type(attachment: AttachmentRecord, text: str) -> str:
    lowered = f"{attachment.filename} {text}".lower()
    if any(token in lowered for token in ("支付", "付款", "扣款", "receipt", "invoice", "refund", "订单", "order")):
        return "payment_screenshot"
    if any(token in lowered for token in ("error", "503", "500", "404", "服务不可用", "登录失败", "mfa", "验证码")):
        return "error_screenshot"
    if attachment.file_type == "pdf" or any(token in lowered for token in ("pdf", "表单", "合同", "文档", "form")):
        return "document_form"
    if attachment.file_type == "log" or any(token in lowered for token in ("exception", "traceback", "timeout", "openstack", "hdfs")):
        return "log_excerpt"
    return "generic_attachment"


def _confidence_for(evidence_type: str, entities: dict[str, object], text: str) -> float:
    if not text.strip():
        return 0.2
    confidence = 0.55
    if evidence_type != "generic_attachment":
        confidence += 0.15
    if entities:
        confidence += min(0.25, len(entities) * 0.08)
    return round(min(confidence, 0.95), 2)


@dataclass(slots=True)
class AttachmentParser:
    """Deterministic V1 parser that turns attachment OCR/summary text into auditable evidence."""

    def parse(self, attachment: AttachmentRecord) -> AttachmentEvidence:
        text = "\n".join(part for part in (attachment.ocr_text, attachment.visual_summary) if part).strip()
        evidence_type = _detect_evidence_type(attachment, text)
        entities: dict[str, object] = {}

        order_id = _first_match(_ORDER_RE, text)
        if order_id:
            entities["order_id"] = order_id.upper().replace("ORDER-", "ORDER-").replace("ORD-", "ORD-")

        amount = _first_match(_AMOUNT_RE, text)
        if amount:
            entities["amount"] = float(amount)

        error_code = _first_match(_ERROR_RE, text)
        if error_code:
            entities["error_code"] = error_code.upper().replace("ERROR ", "")

        risk_flags: list[str] = []
        confidence = _confidence_for(evidence_type, entities, text)
        if confidence < 0.5:
            risk_flags.append("low_confidence_attachment")

        return AttachmentEvidence(
            evidence_id=f"attachment:{attachment.attachment_id}",
            attachment_id=attachment.attachment_id,
            ticket_id=attachment.ticket_id,
            evidence_type=evidence_type,
            extracted_text=attachment.ocr_text,
            visual_summary=attachment.visual_summary,
            entities=entities,
            confidence=confidence,
            source_span=text[:500] if text else None,
            risk_flags=risk_flags,
            metadata={
                "filename": attachment.filename,
                "file_type": attachment.file_type,
                "source_dataset": attachment.source_dataset,
                "content_hash": attachment.content_hash,
                **attachment.metadata,
            },
        )


def attachment_evidence_to_doc(evidence: AttachmentEvidence) -> RetrievedDoc:
    title = f"附件证据：{evidence.evidence_type}"
    snippet_parts = []
    if evidence.visual_summary:
        snippet_parts.append(evidence.visual_summary)
    if evidence.extracted_text:
        snippet_parts.append(evidence.extracted_text)
    if evidence.entities:
        snippet_parts.append("结构化字段=" + ", ".join(f"{key}:{value}" for key, value in evidence.entities.items()))
    snippet = "；".join(snippet_parts) or "附件已接入，但未抽取到可靠文本。"
    return RetrievedDoc(
        doc_id=evidence.evidence_id,
        source_type="attachment",
        title=title,
        snippet=snippet,
        score=evidence.confidence,
        rerank_score=evidence.confidence,
        metadata=evidence.model_dump(mode="json"),
        citations=[
            Citation(
                source_path=f"attachment:{evidence.attachment_id}",
                note=f"附件解析证据，置信度={evidence.confidence}",
            )
        ],
    )
