from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    AutoModel = None
    AutoTokenizer = None

from .models import TicketRecord


CATEGORY_PROTOTYPES: dict[str, list[str]] = {
    "billing_refund": ["退款", "账单扣费", "付款后无法使用", "发票和退款申请"],
    "delivery_issue": ["物流延迟", "订单未发货", "包裹签收异常", "配送进度查询"],
    "technical_issue": ["系统故障", "服务不可用", "报错", "生产环境中断"],
    "account_access": ["无法登录", "账号访问", "密码重置", "验证码和 MFA"],
    "general_inquiry": ["咨询使用方法", "功能说明", "普通问题", "帮助文档"],
}


@dataclass(slots=True)
class IntentPrediction:
    category: str
    confidence: float
    scores: dict[str, float]
    model_name: str


class BertIntentRecognizer:
    """Prototype-based BERT intent recognizer for ticket category hints.

    It does not replace the workflow decision layer. It supplies a semantic
    category hint to the LLM/route layer and can act as a safe fallback when no
    model backend is configured.
    """

    def __init__(self, model_name: str = "bert-base-chinese", device: str | None = None):
        if AutoModel is None or AutoTokenizer is None or torch is None or F is None:
            raise RuntimeError("transformers or torch is not installed")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.prototype_vectors = self._encode_prototypes()

    def predict(self, ticket: TicketRecord) -> IntentPrediction:
        text = f"{ticket.title}\n{ticket.body}\n客户等级={ticket.customer_tier}\n产品={ticket.product}"
        query_vec = self._encode([text])[0]
        scores = {
            category: float((query_vec @ prototype_vec).item())
            for category, prototype_vec in self.prototype_vectors.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_category, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = max(0.0, min(1.0, 0.55 + (best_score - second_score)))
        return IntentPrediction(
            category=best_category,
            confidence=round(confidence, 3),
            scores={key: round(value, 4) for key, value in scores.items()},
            model_name=self.model_name,
        )

    def _encode_prototypes(self) -> dict[str, Any]:
        return {
            category: self._encode(["；".join(examples)])[0]
            for category, examples in CATEGORY_PROTOTYPES.items()
        }

    def _encode(self, texts: list[str]) -> Any:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return F.normalize(pooled, p=2, dim=1)
