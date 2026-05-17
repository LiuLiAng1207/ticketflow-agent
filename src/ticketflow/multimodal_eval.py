from __future__ import annotations

from typing import Any


def _get_attr_or_key(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def score_multimodal_state(state: dict[str, Any], *, expected_evidence_types: set[str] | None = None) -> dict[str, float]:
    """Score one multimodal ticket state without inventing business outcome labels."""

    evidence = list(state.get("attachment_evidence", []) or [])
    docs = list(state.get("retrieved_docs", []) or [])
    expected_evidence_types = expected_evidence_types or set()

    observed_types = {_get_attr_or_key(item, "evidence_type") for item in evidence}
    if expected_evidence_types:
        recall = len(expected_evidence_types & observed_types) / max(len(expected_evidence_types), 1)
    else:
        recall = 1.0 if evidence else 0.0

    attachment_mrr = 0.0
    for rank, doc in enumerate(docs, start=1):
        if _get_attr_or_key(doc, "source_type") == "attachment":
            attachment_mrr = 1.0 / rank
            break

    low_confidence_count = sum(1 for item in evidence if float(_get_attr_or_key(item, "confidence", 0.0) or 0.0) < 0.5)

    return {
        "attachment_parse_success": 1.0 if evidence else 0.0,
        "multimodal_evidence_recall": round(recall, 3),
        "attachment_evidence_mrr": round(attachment_mrr, 3),
        "low_confidence_attachment_rate": round(low_confidence_count / max(len(evidence), 1), 3),
    }
