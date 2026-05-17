from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_local_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Respect explicit environment overrides even if the override is an empty string.
        if key and key not in os.environ:
            os.environ[key] = value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_OPENAI_COMPAT_CHAT_PATH = "/chat/completions"


@dataclass(slots=True)
class TicketFlowSettings:
    project_root: Path
    db_path: Path
    checkpoint_backend: str
    checkpoint_db_path: Path
    seed_dir: Path
    rag_enabled: bool
    rag_embed_backend: str
    rag_embed_model: str
    rag_embed_device: str | None
    rag_db_dir: Path
    rag_top_k: int
    rag_bm25_k1: float
    rag_bm25_b: float
    rag_rrf_k: int
    rag_reranker_backend: str
    rag_reranker_model: str
    rag_reranker_device: str | None
    rag_cache_enabled: bool
    rag_cache_ttl_seconds: int
    redis_url: str | None
    intent_backend: str
    bert_intent_model: str
    bert_intent_device: str | None
    sla_risk_review_enabled: bool
    sla_risk_threshold: float
    reply_review_enabled: bool
    reply_supported_claim_threshold: float
    model_backend: str
    cloud_model_name: str | None = None
    cloud_api_base_url: str | None = None
    cloud_api_key: str | None = None
    cloud_api_chat_path: str = "/chat/completions"
    cloud_api_ca_cert: str | None = None
    minimind_base_url: str | None = None
    minimind_model: str | None = None
    minimind_api_key: str = "sk-local"
    minimind_chat_path: str = "/chat/completions"
    minimind_enable_triage: bool = False
    minimind_enable_draft_reply: bool = True
    minimind_structured_base_url: str | None = None
    minimind_structured_model: str | None = None
    minimind_structured_api_key: str = "sk-local"
    minimind_reply_base_url: str | None = None
    minimind_reply_model: str | None = None
    minimind_reply_api_key: str = "sk-local"
    smtp_host: str | None = None
    smtp_port: int = 465
    smtp_username: str | None = None
    smtp_auth_code: str | None = None
    smtp_use_tls: bool = True
    email_from_name: str = "TicketFlow Agent"
    incident_email_to: str | None = None
    kb_ops_email_to: str | None = None
    enable_production_services: bool = False

    @property
    def cloud_llm_enabled(self) -> bool:
        return bool(self.cloud_model_name and self.cloud_api_base_url and self.cloud_api_key)

    @property
    def minimind_enabled(self) -> bool:
        return bool(self.minimind_model and self.minimind_base_url)

    @property
    def minimind_split_enabled(self) -> bool:
        return bool(
            self.minimind_structured_model
            and self.minimind_structured_base_url
            and self.minimind_reply_model
            and self.minimind_reply_base_url
        )

    @property
    def llm_provider_preset(self) -> bool:
        return bool(self.cloud_model_name and self.cloud_api_base_url)

    @property
    def llm_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.cloud_api_base_url:
            missing.append("OPENAI_COMPAT_BASE_URL")
        if not self.cloud_api_key:
            missing.append("OPENAI_COMPAT_API_KEY")
        if not self.cloud_model_name:
            missing.append("OPENAI_COMPAT_MODEL")
        return missing

    @property
    def llm_enabled(self) -> bool:
        if self.model_backend == "minimind_split":
            return self.minimind_split_enabled or self.cloud_llm_enabled
        if self.model_backend == "minimind_api":
            return self.minimind_enabled or self.cloud_llm_enabled
        if self.model_backend == "cloud_api":
            return self.cloud_llm_enabled
        return False

    @classmethod
    def from_project_root(cls, project_root: Path) -> "TicketFlowSettings":
        _load_local_env(project_root)
        data_root = project_root / "data"
        return cls(
            project_root=project_root,
            db_path=data_root / "generated" / "ticketflow.sqlite3",
            checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "sqlite"),
            checkpoint_db_path=Path(os.getenv("CHECKPOINT_DB_PATH", str(data_root / "generated" / "ticketflow_checkpoints.sqlite3"))),
            seed_dir=data_root / "seed",
            rag_enabled=_get_bool("RAG_ENABLED", True),
            rag_embed_backend=os.getenv("RAG_EMBED_BACKEND", "auto"),
            rag_embed_model=os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
            rag_embed_device=os.getenv("RAG_EMBED_DEVICE"),
            rag_db_dir=Path(os.getenv("RAG_DB_DIR", str(data_root / "generated" / "rag"))),
            rag_top_k=int(os.getenv("RAG_TOP_K", "6")),
            rag_bm25_k1=float(os.getenv("RAG_BM25_K1", "1.5")),
            rag_bm25_b=float(os.getenv("RAG_BM25_B", "0.75")),
            rag_rrf_k=int(os.getenv("RAG_RRF_K", "60")),
            rag_reranker_backend=os.getenv("RAG_RERANKER_BACKEND", "feature"),
            rag_reranker_model=os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            rag_reranker_device=os.getenv("RAG_RERANKER_DEVICE"),
            rag_cache_enabled=_get_bool("RAG_CACHE_ENABLED", False),
            rag_cache_ttl_seconds=int(os.getenv("RAG_CACHE_TTL_SECONDS", "600")),
            redis_url=os.getenv("REDIS_URL"),
            intent_backend=os.getenv("INTENT_BACKEND", "off"),
            bert_intent_model=os.getenv("BERT_INTENT_MODEL", "bert-base-chinese"),
            bert_intent_device=os.getenv("BERT_INTENT_DEVICE"),
            sla_risk_review_enabled=_get_bool("SLA_RISK_REVIEW_ENABLED", True),
            sla_risk_threshold=float(os.getenv("SLA_RISK_THRESHOLD", "0.72")),
            reply_review_enabled=_get_bool("REPLY_REVIEW_ENABLED", True),
            reply_supported_claim_threshold=float(os.getenv("REPLY_SUPPORTED_CLAIM_THRESHOLD", "0.7")),
            model_backend=os.getenv("MODEL_BACKEND", "cloud_api"),
            cloud_model_name=(
                os.getenv("OPENAI_COMPAT_MODEL")
                or os.getenv("DEEPSEEK_MODEL")
                or (DEFAULT_DEEPSEEK_MODEL if os.getenv("DEEPSEEK_API_KEY") else None)
            ),
            cloud_api_base_url=(
                os.getenv("OPENAI_COMPAT_BASE_URL")
                or os.getenv("DEEPSEEK_BASE_URL")
                or (DEFAULT_DEEPSEEK_BASE_URL if os.getenv("DEEPSEEK_API_KEY") else None)
            ),
            cloud_api_key=os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            cloud_api_chat_path=os.getenv("OPENAI_COMPAT_CHAT_PATH", DEFAULT_OPENAI_COMPAT_CHAT_PATH),
            cloud_api_ca_cert=os.getenv("OPENAI_COMPAT_CA_CERT"),
            minimind_base_url=os.getenv("MINIMIND_BASE_URL"),
            minimind_model=os.getenv("MINIMIND_MODEL"),
            minimind_api_key=os.getenv("MINIMIND_API_KEY", "sk-local"),
            minimind_chat_path=os.getenv("MINIMIND_CHAT_PATH", "/chat/completions"),
            minimind_enable_triage=_get_bool("MINIMIND_ENABLE_TRIAGE", False),
            minimind_enable_draft_reply=_get_bool("MINIMIND_ENABLE_DRAFT_REPLY", True),
            minimind_structured_base_url=os.getenv("MINIMIND_STRUCTURED_BASE_URL"),
            minimind_structured_model=os.getenv("MINIMIND_STRUCTURED_MODEL"),
            minimind_structured_api_key=os.getenv("MINIMIND_STRUCTURED_API_KEY", os.getenv("MINIMIND_API_KEY", "sk-local")),
            minimind_reply_base_url=os.getenv("MINIMIND_REPLY_BASE_URL"),
            minimind_reply_model=os.getenv("MINIMIND_REPLY_MODEL"),
            minimind_reply_api_key=os.getenv("MINIMIND_REPLY_API_KEY", os.getenv("MINIMIND_API_KEY", "sk-local")),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "465")),
            smtp_username=os.getenv("SMTP_USERNAME"),
            smtp_auth_code=os.getenv("SMTP_AUTH_CODE"),
            smtp_use_tls=_get_bool("SMTP_USE_TLS", True),
            email_from_name=os.getenv("EMAIL_FROM_NAME", "TicketFlow Agent"),
            incident_email_to=os.getenv("INCIDENT_EMAIL_TO"),
            kb_ops_email_to=os.getenv("KB_OPS_EMAIL_TO"),
            enable_production_services=_get_bool("ENABLE_PRODUCTION_SERVICES", False),
        )
