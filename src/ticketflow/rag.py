from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .db import TicketFlowRepository, _query_terms
from .models import Citation, RetrievedDoc, RetrievalStats, TicketRecord

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    AutoModel = None
    AutoTokenizer = None

try:
    from FlagEmbedding import FlagReranker
except ImportError:  # pragma: no cover - optional dependency
    FlagReranker = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


def _normalize_text(text: str) -> str:
    return " ".join(_query_terms(text))


def _category_keywords(ticket: TicketRecord) -> list[str]:
    mapping = {
        "billing_refund": ["refund", "billing", "payment", "chargeback"],
        "delivery_issue": ["delivery", "shipping", "package", "tracking"],
        "technical_issue": ["incident", "outage", "error", "technical"],
        "account_access": ["login", "account", "access", "password"],
        "general_inquiry": ["question", "help", "guide", "general"],
    }
    category = ticket.expected_category or "general_inquiry"
    return mapping.get(category, mapping["general_inquiry"])


def reciprocal_rank_fusion_score(
    *,
    bm25_rank: int | None = None,
    vector_rank: int | None = None,
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> float:
    """Weighted RRF score. Ranks are 1-based; missing ranks contribute 0."""
    score = 0.0
    if bm25_rank is not None:
        score += bm25_weight / (k + bm25_rank)
    if vector_rank is not None:
        score += vector_weight / (k + vector_rank)
    return score


@dataclass(slots=True)
class BM25Hit:
    doc_id: str
    source_type: str
    title: str
    snippet: str
    document: str
    score: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class BM25CorpusIndex:
    documents: list[dict[str, Any]]
    tokenized_documents: list[list[str]]
    idf: dict[str, float]
    avg_doc_len: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_documents(cls, documents: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> "BM25CorpusIndex":
        tokenized_documents = [_query_terms(str(item.get("document") or "")) for item in documents]
        doc_count = max(len(tokenized_documents), 1)
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_documents:
            doc_freq.update(set(tokens))
        idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        avg_doc_len = sum(len(tokens) for tokens in tokenized_documents) / doc_count if tokenized_documents else 0.0
        return cls(
            documents=documents,
            tokenized_documents=tokenized_documents,
            idf=idf,
            avg_doc_len=max(avg_doc_len, 1.0),
            k1=k1,
            b=b,
        )

    def search(self, query: str, *, limit: int = 10) -> list[BM25Hit]:
        query_terms = _query_terms(query)
        if not query_terms:
            return []
        scored: list[BM25Hit] = []
        for item, tokens in zip(self.documents, self.tokenized_documents, strict=False):
            score = self._score_tokens(query_terms, tokens)
            if score <= 0:
                continue
            metadata = dict(item.get("metadata") or {})
            scored.append(
                BM25Hit(
                    doc_id=str(metadata.get("doc_id") or item.get("id") or ""),
                    source_type=str(metadata.get("source_type") or "kb"),
                    title=str(metadata.get("title") or ""),
                    snippet=str(metadata.get("snippet") or item.get("document") or ""),
                    document=str(item.get("document") or ""),
                    score=score,
                    metadata=metadata,
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def _score_tokens(self, query_terms: list[str], doc_terms: list[str]) -> float:
        if not doc_terms:
            return 0.0
        term_counts = Counter(doc_terms)
        doc_len = len(doc_terms)
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (tf * (self.k1 + 1.0)) / max(denominator, 1e-9)
        return score


class RetrievalCache:
    def __init__(self, *, redis_url: str | None, ttl_seconds: int = 600):
        self.ttl_seconds = ttl_seconds
        self.backend = "memory"
        self.failure_reason: str | None = None
        self._memory: dict[str, tuple[float, dict[str, Any]]] = {}
        self._redis_client: Any | None = None
        if redis_url:
            if redis is None:
                self.failure_reason = "redis-py is not installed; using in-memory cache."
                return
            try:
                self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis_client.ping()
                self.backend = "redis"
            except Exception as exc:  # pragma: no cover - external service dependent
                self._redis_client = None
                self.failure_reason = f"Redis unavailable: {exc}; using in-memory cache."

    def get(self, key: str) -> dict[str, Any] | None:
        if self._redis_client is not None:
            raw = self._redis_client.get(key)
            return json.loads(raw) if raw else None
        item = self._memory.get(key)
        if item is None:
            return None
        expires_at, payload = item
        if expires_at < time.time():
            self._memory.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: dict[str, Any]) -> None:
        if self._redis_client is not None:
            self._redis_client.setex(key, self.ttl_seconds, json.dumps(payload, ensure_ascii=False))
            return
        self._memory[key] = (time.time() + self.ttl_seconds, payload)


_HF_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}


class HashEmbeddingFunction:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def name(self) -> str:
        return "ticketflow-hash-embedding"

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            input = [input]
        return [self._embed(text) for text in input]

    def embed_documents(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        return self(input)

    def embed_query(self, input: list[str] | str) -> list[float]:  # noqa: A002
        if isinstance(input, list):
            input = " ".join(str(item) for item in input)
        return self._embed(input)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for term in _query_terms(text):
            if not term:
                continue
            digest = hashlib.sha256(term.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0 + min(len(term), 4) * 0.1
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class TransformersEmbeddingFunction:
    def __init__(self, model_name: str, device: str | None = None):
        if AutoModel is None or AutoTokenizer is None or torch is None or F is None:
            raise RuntimeError("transformers or torch is not installed")
        self.model_name = model_name
        self.device = device
        cache_key = (model_name, device)
        if cache_key not in _HF_MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(resolved_device)
            model.eval()
            _HF_MODEL_CACHE[cache_key] = {
                "tokenizer": tokenizer,
                "model": model,
                "device": resolved_device,
            }
        cached = _HF_MODEL_CACHE[cache_key]
        self.tokenizer = cached["tokenizer"]
        self.model = cached["model"]
        self.device = cached["device"]
        self.dimensions = int(getattr(self.model.config, "hidden_size", 768))

    def name(self) -> str:
        return f"transformers:{self.model_name}"

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            input = [input]
        return self._encode(input)

    def embed_documents(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            input = [input]
        return self._encode(input)

    def embed_query(self, input: list[str] | str) -> list[float]:  # noqa: A002
        if isinstance(input, list):
            input = " ".join(str(item) for item in input)
        return self._encode([input], is_query=True)[0]

    def _encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prepared = texts
        model_name = self.model_name.lower()
        if "bge" in self.model_name.lower():
            prepared = [f"为这个句子生成表示以用于检索相关文章：{text}" for text in texts]
        if "bge-m3" in model_name:
            prepared = texts
        elif "bge" in model_name:
            prepared = [f"为这个句子生成表示以用于检索相关文章：{text}" for text in texts] if is_query else texts
        encoded = self.tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            if self.model_name.lower().endswith("bge-m3"):
                sentence_embeddings = token_embeddings[:, 0]
            else:
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                sentence_embeddings = pooled / counts
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().tolist()


@dataclass(slots=True)
class RagSearchResult:
    docs: list[RetrievedDoc]
    stats: RetrievalStats


@dataclass(slots=True)
class HybridRetriever:
    repository: TicketFlowRepository
    rag_db_dir: Path
    top_k: int = 6
    enabled: bool = True
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    embed_backend: str = "auto"
    embed_device: str | None = None
    embedding_fn: Any | None = None
    collection: Any | None = None
    embedding_backend_name: str = "hash"
    embedding_failure_reason: str | None = None
    collection_name: str = "ticketflow_rag"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    reranker_backend: str = "feature"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = None
    cache_enabled: bool = False
    redis_url: str | None = None
    cache_ttl_seconds: int = 600
    corpus_cache: list[dict[str, Any]] = field(default_factory=list)
    bm25_index: BM25CorpusIndex | None = None
    bge_reranker: Any | None = None
    bge_reranker_impl: str | None = None
    reranker_failure_reason: str | None = None
    retrieval_cache: RetrievalCache | None = None

    def __post_init__(self) -> None:
        self.rag_db_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = self._build_embedding_function()
        self.collection_name = self._build_collection_name()
        self.corpus_cache = self._corpus_documents()
        self.bm25_index = BM25CorpusIndex.from_documents(
            self.corpus_cache,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )
        self.bge_reranker = self._build_bge_reranker()
        if self.cache_enabled:
            self.retrieval_cache = RetrievalCache(redis_url=self.redis_url, ttl_seconds=self.cache_ttl_seconds)
        if self.enabled and chromadb is not None:
            client = chromadb.PersistentClient(path=str(self.rag_db_dir))
            self._maybe_reset_collection(client)
            self.collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={
                    "source": "ticketflow",
                    "embed_model": self.embed_model,
                    "embed_backend": self.embedding_backend_name,
                    "embed_dimensions": self._embedding_dimensions(),
                    "hnsw:space": "cosine",
                },
            )
            self._sync_collection()

    def _build_embedding_function(self) -> Any:
        backend = (self.embed_backend or "auto").strip().lower()
        if backend in {"auto", "sentence_transformers", "sentence-transformers", "st"}:
            backend = "auto" if backend == "auto" else "transformers"
        if backend in {"auto", "transformers", "hf", "huggingface"}:
            try:
                embedding_fn = TransformersEmbeddingFunction(
                    model_name=self.embed_model,
                    device=self.embed_device,
                )
                self.embedding_backend_name = "transformers"
                self.embedding_failure_reason = None
                return embedding_fn
            except Exception as exc:  # pragma: no cover - environment dependent
                self.embedding_failure_reason = f"transformers embedding unavailable: {exc}"
                if backend != "auto":
                    raise
        self.embedding_backend_name = "hash"
        return HashEmbeddingFunction()

    def _manifest_path(self) -> Path:
        return self.rag_db_dir / "manifest.json"

    def _embedding_dimensions(self) -> int:
        return int(getattr(self.embedding_fn, "dimensions", 0) or 0)

    def _embedding_signature(self) -> str:
        raw = f"{self.embedding_backend_name}|{self.embed_model}|{self._embedding_dimensions()}|bge_prompt_v2"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _build_collection_name(self) -> str:
        return f"ticketflow_rag_{self._embedding_signature()}"

    def _load_manifest(self) -> dict[str, Any] | None:
        path = self._manifest_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _maybe_reset_collection(self, client: Any) -> None:
        manifest = self._load_manifest()
        previous_collection_name = None
        if manifest:
            previous_collection_name = manifest.get("collection_name")
            previous_backend = manifest.get("embed_backend_used")
            previous_model = manifest.get("embed_model")
            previous_dimensions = int(manifest.get("embed_dimensions") or 0)
            if (
                previous_collection_name == self.collection_name
                and previous_backend == self.embedding_backend_name
                and previous_model == self.embed_model
                and previous_dimensions == self._embedding_dimensions()
            ):
                return
        cleanup_candidates = {
            "ticketflow_rag",  # legacy fixed-name collection
        }
        if previous_collection_name:
            cleanup_candidates.add(str(previous_collection_name))
        for name in cleanup_candidates:
            if name == self.collection_name:
                continue
            try:
                client.delete_collection(name)
            except Exception:
                pass

    def _corpus_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for row in self.repository.list_kb_articles():
            documents.append(
                {
                    "id": f"kb:{row['doc_id']}",
                    "document": f"{row['title']}\n{row['body']}\n{row['tags']}",
                    "metadata": {
                        "source_type": "kb",
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "product": row["product"],
                        "category": row["category"],
                        "snippet": row["body"],
                    },
                }
            )
        for row in self.repository.list_policies():
            documents.append(
                {
                    "id": f"policy:{row['policy_id']}",
                    "document": f"{row['title']}\n{row['body']}\n{row['action_type']}",
                    "metadata": {
                        "source_type": "policy",
                        "doc_id": row["policy_id"],
                        "title": row["title"],
                        "action_type": row["action_type"],
                        "approval_required": bool(row["approval_required"]),
                        "priority_hint": row["priority_hint"],
                        "snippet": row["body"],
                    },
                }
            )
        for row in self.repository.list_ticket_history_entries():
            ticket = self.repository.get_ticket(str(row["ticket_id"]))
            history_text = "\n".join(
                filter(
                    None,
                    [
                        str(row["message"]),
                        str(row["agent_name"]),
                        ticket.title,
                        ticket.body,
                        ticket.product,
                        ticket.expected_category or "",
                    ],
                )
            )
            documents.append(
                {
                    "id": f"history:{row['event_id']}",
                    "document": history_text,
                    "metadata": {
                        "source_type": "history",
                        "doc_id": row["event_id"],
                        "ticket_id": row["ticket_id"],
                        "title": ticket.title,
                        "product": ticket.product,
                        "category": ticket.expected_category,
                        "agent_name": row["agent_name"],
                        "snippet": row["message"],
                    },
                }
            )
        return documents

    def _sync_collection(self) -> None:
        if self.collection is None:
            return
        documents = self.corpus_cache or self._corpus_documents()
        if not documents:
            return
        ids = [doc["id"] for doc in documents]
        payloads = [doc["document"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        self.collection.upsert(ids=ids, documents=payloads, metadatas=metadatas)

    def retrieve(self, ticket: TicketRecord) -> RagSearchResult:
        query = f"{ticket.title}\n{ticket.body}\n{ticket.product}\n{' '.join(_category_keywords(ticket))}"
        cache_key = self._cache_key(ticket, query)
        if self.retrieval_cache is not None:
            cached = self.retrieval_cache.get(cache_key)
            if cached is not None:
                docs = [RetrievedDoc.model_validate(item) for item in cached.get("docs", [])]
                stats = RetrievalStats.model_validate(cached.get("stats", {}))
                stats.metadata["cache_hit"] = True
                return RagSearchResult(docs=docs, stats=stats)
        lexical_docs = self._lexical_search(ticket, query)
        vector_hits = 0
        vector_docs: list[RetrievedDoc] = []
        if self.collection is not None:
            vector_docs = self._vector_search(query, self.top_k * 2)
            vector_hits = len(vector_docs)
        merged_docs = self._merge_retrievals(lexical_docs, vector_docs)
        reranked_docs = self._rerank_docs(ticket, list(merged_docs.values()))
        final_docs = self._select_balanced_docs(reranked_docs, ticket=ticket)
        source_counts = Counter(doc.source_type for doc in final_docs)
        stats = RetrievalStats(
            lexical_hits=len(lexical_docs),
            vector_hits=vector_hits,
            final_hits=len(final_docs),
            used_vector=vector_hits > 0,
            fallback_to_keywords=self.collection is None,
            source_counts=dict(source_counts),
            retrieved_doc_ids=[f"{doc.source_type}:{doc.doc_id}" for doc in final_docs],
            metadata={
                "lexical_backend": "bm25",
                "fusion_method": "rrf",
                "rrf_k": self.rrf_k,
                "embedding_backend": self.embedding_backend_name,
                "embedding_model": self.embed_model,
                "reranker_backend": self._effective_reranker_backend(),
                "reranker_model": self.reranker_model if self._effective_reranker_backend() == "bge-reranker-v2-m3" else None,
                "reranker_impl": self.bge_reranker_impl,
                "reranker_failure_reason": self.reranker_failure_reason,
                "cache_backend": self.retrieval_cache.backend if self.retrieval_cache is not None else None,
                "cache_hit": False,
                "cache_failure_reason": self.retrieval_cache.failure_reason if self.retrieval_cache is not None else None,
                "source_recall_lanes": ["product_kb_keyword"],
            },
        )
        result = RagSearchResult(docs=final_docs, stats=stats)
        if self.retrieval_cache is not None:
            self.retrieval_cache.set(
                cache_key,
                {
                    "docs": [doc.model_dump(mode="json") for doc in final_docs],
                    "stats": stats.model_dump(mode="json"),
                },
            )
        return result

    def _cache_key(self, ticket: TicketRecord, query: str) -> str:
        raw = json.dumps(
            {
                "ticket_id": ticket.ticket_id,
                "query": query,
                "top_k": self.top_k,
                "embed": self._embedding_signature(),
                "bm25_k1": self.bm25_k1,
                "bm25_b": self.bm25_b,
                "rrf_k": self.rrf_k,
                "reranker": self._effective_reranker_backend(),
                "source_recall_lanes": "product_kb_keyword_v2",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"ticketflow:rag:{hashlib.sha1(raw.encode('utf-8')).hexdigest()}"

    def _merge_retrievals(
        self,
        lexical_docs: list[RetrievedDoc],
        vector_docs: list[RetrievedDoc],
    ) -> dict[tuple[str, str], RetrievedDoc]:
        lexical_weight, vector_weight = self._fusion_weights()
        lexical_ranks = {
            (doc.source_type, doc.doc_id): index + 1
            for index, doc in enumerate(sorted(lexical_docs, key=lambda item: item.score, reverse=True))
        }
        vector_ranks = {
            (doc.source_type, doc.doc_id): index + 1
            for index, doc in enumerate(sorted(vector_docs, key=lambda item: item.score, reverse=True))
        }
        lexical_map = {(doc.source_type, doc.doc_id): doc for doc in lexical_docs}
        vector_map = {(doc.source_type, doc.doc_id): doc for doc in vector_docs}
        all_keys = set(lexical_map) | set(vector_map)
        merged: dict[tuple[str, str], RetrievedDoc] = {}
        for key in all_keys:
            lexical_doc = lexical_map.get(key)
            vector_doc = vector_map.get(key)
            base_doc = lexical_doc or vector_doc
            if base_doc is None:
                continue
            lexical_rank = lexical_ranks.get(key)
            vector_rank = vector_ranks.get(key)
            semantic_score = vector_doc.score if vector_doc is not None else 0.0
            fused_score = reciprocal_rank_fusion_score(
                bm25_rank=lexical_rank,
                vector_rank=vector_rank,
                k=self.rrf_k,
                bm25_weight=lexical_weight,
                vector_weight=vector_weight,
            )
            metadata = dict(base_doc.metadata)
            metadata.update(
                {
                    "bm25_rank": lexical_rank,
                    "vector_rank": vector_rank,
                    "bm25_score": round(float(lexical_doc.score), 4) if lexical_doc is not None else 0.0,
                    "vector_semantic_score": round(semantic_score, 4),
                    "rrf_score": round(fused_score, 6),
                    "retrieval": "bm25_vector_rrf",
                }
            )
            merged[key] = base_doc.model_copy(
                update={
                    "score": fused_score,
                    "rerank_score": fused_score,
                    "metadata": metadata,
                }
            )
        return merged

    @staticmethod
    def _normalized_overlap(query_terms: set[str], candidate_terms: set[str]) -> float:
        if not query_terms or not candidate_terms:
            return 0.0
        return len(query_terms & candidate_terms) / len(query_terms)

    def _doc_rerank_features(self, ticket: TicketRecord, doc: RetrievedDoc) -> dict[str, float]:
        query_terms = set(_query_terms(f"{ticket.title} {ticket.body} {ticket.product}"))
        title_terms = set(_query_terms(doc.title))
        snippet_terms = set(_query_terms(doc.snippet))
        metadata_text = " ".join(str(value or "") for value in doc.metadata.values())
        metadata_terms = set(_query_terms(metadata_text))
        title_focus_terms = set(_query_terms(ticket.title))

        text_overlap = self._normalized_overlap(query_terms, title_terms | snippet_terms | metadata_terms)
        title_overlap = self._normalized_overlap(title_focus_terms or query_terms, title_terms)
        metadata_overlap = self._normalized_overlap(query_terms, metadata_terms)
        exact_product_match = float(
            bool(ticket.product)
            and str(doc.metadata.get("product") or "").strip().lower() == str(ticket.product).strip().lower()
        )
        exact_ticket_match = float(
            doc.source_type == "history"
            and bool(ticket.ticket_id)
            and str(doc.metadata.get("ticket_id") or "").strip().lower() == str(ticket.ticket_id).strip().lower()
        )
        category_match = float(
            bool(ticket.expected_category)
            and str(doc.metadata.get("category") or "").strip().lower() == str(ticket.expected_category).strip().lower()
        )
        policy_action_match = float(
            doc.source_type == "policy"
            and ticket.expected_category == "billing_refund"
            and str(doc.metadata.get("action_type") or "").strip().lower() == "refund"
        )
        policy_text = f"{doc.title} {doc.snippet}".lower()
        missing_order_policy_match = float(
            doc.source_type == "policy"
            and ticket.expected_category == "billing_refund"
            and not ticket.linked_order_id
            and any(token in policy_text for token in ("订单", "order"))
            and any(token in policy_text for token in ("缺少", "补充", "核验", "missing", "verify"))
        )
        source_is_history = float(doc.source_type == "history")
        return {
            "text_overlap": round(text_overlap, 4),
            "title_overlap": round(title_overlap, 4),
            "metadata_overlap": round(metadata_overlap, 4),
            "exact_product_match": exact_product_match,
            "exact_ticket_match": exact_ticket_match,
            "category_match": category_match,
            "policy_action_match": policy_action_match,
            "missing_order_policy_match": missing_order_policy_match,
            "source_is_history": source_is_history,
        }

    def _rerank_score(self, doc: RetrievedDoc, features: dict[str, float]) -> float:
        base_score = float(doc.score)
        text_overlap = features["text_overlap"]
        title_overlap = features["title_overlap"]
        metadata_overlap = features["metadata_overlap"]
        exact_product_match = features["exact_product_match"]
        exact_ticket_match = features["exact_ticket_match"]
        category_match = features["category_match"]
        policy_action_match = features["policy_action_match"]
        missing_order_policy_match = features["missing_order_policy_match"]

        if doc.source_type == "history":
            rerank_score = (
                0.34 * base_score
                + 0.22 * text_overlap
                + 0.12 * title_overlap
                + 0.07 * metadata_overlap
                + 0.05 * exact_product_match
                + 0.08 * category_match
                + 0.12 * exact_ticket_match
            )
            if exact_ticket_match:
                rerank_score = max(rerank_score, 0.98)
        elif doc.source_type == "policy":
            rerank_score = (
                0.44 * base_score
                + 0.18 * text_overlap
                + 0.08 * title_overlap
                + 0.08 * metadata_overlap
                + 0.12 * policy_action_match
                + 0.10 * missing_order_policy_match
            )
        else:
            rerank_score = (
                0.60 * base_score
                + 0.18 * text_overlap
                + 0.10 * title_overlap
                + 0.07 * metadata_overlap
                + 0.05 * exact_product_match
            )
            if doc.source_type == "kb" and doc.metadata.get("source_specific_rank") is not None:
                source_rank = int(doc.metadata.get("source_specific_rank") or 0)
                if source_rank > 0:
                    # Product-filtered KB retrieval is a high-precision recall
                    # lane; keep its source-internal order visible after semantic
                    # reranking instead of letting generic history matches bury it.
                    rerank_score = max(rerank_score, 0.94 - 0.015 * (source_rank - 1))
        return max(0.0, min(1.0, rerank_score))

    def _rerank_docs(self, ticket: TicketRecord, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if self.bge_reranker is not None and docs:
            query = f"{ticket.title}\n{ticket.body}\n{ticket.product}"
            try:
                pairs = [[query, f"{doc.title}\n{doc.snippet}"] for doc in docs]
                raw_scores = self._compute_bge_rerank_scores(pairs)
                if not isinstance(raw_scores, list):
                    raw_scores = [float(raw_scores)]
                bge_scores = self._minmax_normalize(raw_scores)
                reranked_with_bge: list[RetrievedDoc] = []
                for doc, raw_score, bge_score in zip(docs, raw_scores, bge_scores, strict=False):
                    features = self._doc_rerank_features(ticket, doc)
                    feature_score = self._rerank_score(doc, features)
                    # Cross-encoder semantics should refine, not erase, high-precision BM25/RRF
                    # and metadata signals such as policy/order/history source matches.
                    final_score = max(0.0, min(1.0, 0.95 * feature_score + 0.05 * bge_score))
                    metadata = dict(doc.metadata)
                    metadata.update(
                        {
                            "raw_fused_score": round(float(doc.score), 6),
                            "feature_rerank_score": round(feature_score, 4),
                            "rerank_text_overlap": features["text_overlap"],
                            "rerank_title_overlap": features["title_overlap"],
                            "rerank_metadata_overlap": features["metadata_overlap"],
                            "rerank_exact_product_match": features["exact_product_match"],
                            "rerank_exact_ticket_match": features["exact_ticket_match"],
                            "rerank_category_match": features["category_match"],
                            "rerank_policy_action_match": features["policy_action_match"],
                            "rerank_missing_order_policy_match": features["missing_order_policy_match"],
                            "bge_reranker_raw_score": round(float(raw_score), 4),
                            "bge_reranker_score": round(float(bge_score), 4),
                            "reranker": "BAAI/bge-reranker-v2-m3",
                            "reranker_impl": self.bge_reranker_impl or "unknown",
                            "rerank_strategy": "feature_primary_bge_tiebreak",
                        }
                    )
                    reranked_with_bge.append(
                        doc.model_copy(
                            update={
                                "score": final_score,
                                "rerank_score": final_score,
                                "metadata": metadata,
                            }
                        )
                    )
                return sorted(reranked_with_bge, key=lambda item: item.rerank_score or item.score, reverse=True)
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                self.reranker_failure_reason = f"BGE reranker runtime failed: {exc}"

        reranked: list[RetrievedDoc] = []
        for doc in docs:
            features = self._doc_rerank_features(ticket, doc)
            rerank_score = self._rerank_score(doc, features)
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "raw_fused_score": round(float(doc.score), 4),
                    "rerank_score": round(rerank_score, 4),
                    "rerank_text_overlap": features["text_overlap"],
                    "rerank_title_overlap": features["title_overlap"],
                    "rerank_metadata_overlap": features["metadata_overlap"],
                    "rerank_exact_product_match": features["exact_product_match"],
                    "rerank_exact_ticket_match": features["exact_ticket_match"],
                    "rerank_category_match": features["category_match"],
                    "rerank_policy_action_match": features["policy_action_match"],
                    "rerank_missing_order_policy_match": features["missing_order_policy_match"],
                }
            )
            reranked.append(
                doc.model_copy(
                    update={
                        "score": rerank_score,
                        "rerank_score": rerank_score,
                        "metadata": metadata,
                    }
                )
            )
        return sorted(reranked, key=lambda item: item.rerank_score or item.score, reverse=True)

    def _fusion_weights(self) -> tuple[float, float]:
        if self.embedding_backend_name == "transformers":
            return (1.0, 1.0)
        return (1.15, 0.85)

    def _effective_reranker_backend(self) -> str:
        if self.bge_reranker is not None:
            return "bge-reranker-v2-m3"
        return "feature"

    @staticmethod
    def _minmax_normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(min_score, max_score):
            return [0.5 for _ in scores]
        span = max_score - min_score
        return [max(0.0, min(1.0, (float(score) - min_score) / span)) for score in scores]

    def _compute_bge_rerank_scores(self, pairs: list[list[str]]) -> list[float]:
        if self.bge_reranker_impl == "cross_encoder":
            raw_scores = self.bge_reranker.predict(pairs)
            return [float(score) for score in raw_scores]
        scores = self.bge_reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, list):
            return [float(score) for score in scores]
        return [float(scores)]

    def _build_bge_reranker(self) -> Any | None:
        backend = (self.reranker_backend or "feature").strip().lower()
        if backend not in {"bge", "bge-reranker", "bge-reranker-v2-m3"}:
            return None
        try:
            device = self.reranker_device or ("cuda:0" if torch is not None and torch.cuda.is_available() else "cpu")
            if CrossEncoder is not None:
                self.bge_reranker_impl = "cross_encoder"
                return CrossEncoder(self.reranker_model, device=device)
            if FlagReranker is not None:
                self.bge_reranker_impl = "flag_embedding"
                return FlagReranker(self.reranker_model, devices=[device], use_fp16=device.startswith("cuda"))
            self.reranker_failure_reason = "No BGE reranker runtime is installed; using feature reranker."
            return None
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            self.bge_reranker_impl = None
            self.reranker_failure_reason = f"BGE reranker unavailable: {exc}"
            return None

    def _lexical_search(self, ticket: TicketRecord, query: str) -> list[RetrievedDoc]:
        if self.bm25_index is None:
            return []
        docs = self._source_specific_keyword_search(ticket, query)
        seen_keys = {(doc.source_type, doc.doc_id) for doc in docs}
        hits = self.bm25_index.search(query, limit=max(len(self.corpus_cache), self.top_k * 12, 50))
        for rank, hit in enumerate(hits, start=1):
            key = (hit.source_type, hit.doc_id)
            if key in seen_keys:
                continue
            note = {
                "kb": "Knowledge base",
                "policy": "Policy rule",
                "history": "Historical case",
            }.get(hit.source_type, "BM25 evidence")
            metadata = dict(hit.metadata)
            metadata.update({"retrieval": "bm25", "bm25_rank": rank, "bm25_score": round(hit.score, 4)})
            docs.append(
                RetrievedDoc(
                    doc_id=hit.doc_id,
                    source_type=hit.source_type,  # type: ignore[arg-type]
                    title=hit.title,
                    snippet=hit.snippet,
                    score=hit.score,
                    metadata=metadata,
                    citations=[Citation(source_path=f"{hit.source_type}:{hit.doc_id}", note=note)],
                )
            )
            seen_keys.add(key)
        return docs

    def _source_specific_keyword_search(self, ticket: TicketRecord, query: str) -> list[RetrievedDoc]:
        docs: list[RetrievedDoc] = []
        product_query = f"{ticket.title}\n{ticket.body}\n{ticket.product}"
        for rank, row in enumerate(self.repository.search_kb(product_query, ticket.product, limit=max(3, self.top_k // 2)), start=1):
            score = 1000.0 - rank + float(row.get("score", 0.0))
            docs.append(
                RetrievedDoc(
                    doc_id=str(row["doc_id"]),
                    source_type="kb",
                    title=str(row["title"]),
                    snippet=str(row["body"]),
                    score=score,
                    metadata={
                        "product": row.get("product"),
                        "category": row.get("category"),
                        "retrieval": "product_kb_keyword_lane",
                        "source_specific_rank": rank,
                        "source_specific_score": round(score, 4),
                    },
                    citations=[Citation(source_path=f"kb:{row['doc_id']}", note="Product knowledge base")],
                )
            )
        return docs

    def _vector_search(self, query: str, limit: int) -> list[RetrievedDoc]:
        if self.collection is None:
            return []
        query_embedding = self.embedding_fn.embed_query(query) if self.embedding_fn else []
        raw = self.collection.query(query_embeddings=[query_embedding], n_results=limit)
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(metadatas)
        docs: list[RetrievedDoc] = []
        for idx, metadata in enumerate(metadatas):
            if not metadata:
                continue
            source_type = metadata["source_type"]
            doc_id = metadata["doc_id"]
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            score = max(0.0, min(1.0, 1.0 - distance))
            note = {
                "kb": "Knowledge base",
                "policy": "Policy rule",
                "history": "Historical case",
            }.get(source_type, "Retrieved evidence")
            docs.append(
                RetrievedDoc(
                    doc_id=str(doc_id),
                    source_type=source_type,
                    title=str(metadata["title"]),
                    snippet=str(metadata["snippet"]),
                    score=score,
                    metadata=dict(metadata) | {"retrieval": "vector"},
                    citations=[Citation(source_path=f"{source_type}:{doc_id}", note=note)],
                )
            )
        return docs

    def baseline_retrieve(self, ticket: TicketRecord) -> RagSearchResult:
        docs: list[RetrievedDoc] = []
        for row in self.repository.search_kb(f"{ticket.title}\n{ticket.body}", ticket.product, limit=self.top_k):
            docs.append(
                RetrievedDoc(
                    doc_id=str(row["doc_id"]),
                    source_type="kb",
                    title=str(row["title"]),
                    snippet=str(row["body"]),
                    score=float(row.get("score", 0.0)),
                    metadata={"product": row.get("product"), "category": row.get("category"), "retrieval": "baseline_keyword"},
                    citations=[Citation(source_path=f"kb:{row['doc_id']}", note="Knowledge base")],
                )
            )
        stats = RetrievalStats(
            lexical_hits=len(docs),
            vector_hits=0,
            final_hits=len(docs),
            used_vector=False,
            fallback_to_keywords=True,
            source_counts=dict(Counter(doc.source_type for doc in docs)),
            retrieved_doc_ids=[f"{doc.source_type}:{doc.doc_id}" for doc in docs],
        )
        return RagSearchResult(docs=docs, stats=stats)

    def write_manifest(self) -> None:
        manifest = {
            "enabled": self.enabled,
            "embed_backend_requested": self.embed_backend,
            "embed_backend_used": self.embedding_backend_name,
            "embed_model": self.embed_model,
            "embed_dimensions": self._embedding_dimensions(),
            "embed_device": self.embed_device,
            "embedding_failure_reason": self.embedding_failure_reason,
            "backend": "chromadb" if self.collection is not None else "keyword_only",
            "collection_name": self.collection_name,
            "lexical_backend": "bm25",
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "fusion_method": "rrf",
            "rrf_k": self.rrf_k,
            "reranker_backend": self._effective_reranker_backend(),
            "reranker_model": self.reranker_model if self._effective_reranker_backend() == "bge-reranker-v2-m3" else None,
            "reranker_impl": self.bge_reranker_impl,
            "reranker_failure_reason": self.reranker_failure_reason,
            "doc_count": len(self.corpus_cache or self._corpus_documents()),
        }
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _select_balanced_docs(self, docs: list[RetrievedDoc], *, ticket: TicketRecord | None = None) -> list[RetrievedDoc]:
        ranked = sorted(docs, key=lambda item: item.rerank_score or item.score, reverse=True)
        selected = ranked[: self.top_k]
        selected_keys = {(doc.source_type, doc.doc_id) for doc in selected}

        def replace_at(index: int, candidate: RetrievedDoc) -> None:
            selected_keys.discard((selected[index].source_type, selected[index].doc_id))
            selected[index] = candidate
            selected_keys.add((candidate.source_type, candidate.doc_id))

        def add_or_replace(candidate: RetrievedDoc) -> None:
            candidate_key = (candidate.source_type, candidate.doc_id)
            if candidate_key in selected_keys:
                return
            if len(selected) < self.top_k:
                selected.append(candidate)
                selected_keys.add(candidate_key)
                return

            source_counts = Counter(doc.source_type for doc in selected)
            replacement_index = None
            for index in range(len(selected) - 1, -1, -1):
                if source_counts[selected[index].source_type] > 1:
                    replacement_index = index
                    break
            if replacement_index is None:
                replacement_index = len(selected) - 1
            replace_at(replacement_index, candidate)

        def policy_fingerprint(doc: RetrievedDoc) -> str:
            if doc.source_type != "policy":
                return ""
            return _normalize_text(f"{doc.title} {doc.snippet}") or str(doc.title).strip().lower()

        # Keep the final context relevance-ordered, but avoid dropping an entire
        # evidence family when source diversity can be recovered by replacing a
        # duplicate low-scoring source.
        for source in ("policy", "kb", "history"):
            if any(doc.source_type == source for doc in selected):
                continue
            source_candidate = next((doc for doc in ranked if doc.source_type == source), None)
            if source_candidate is None:
                continue
            candidate_key = (source_candidate.source_type, source_candidate.doc_id)
            if candidate_key in selected_keys:
                continue
            add_or_replace(source_candidate)

        if ticket is not None and ticket.expected_category in {"billing_refund", "delivery_issue", "technical_issue", "account_access"}:
            min_policy_docs = 2
            while len({policy_fingerprint(doc) for doc in selected if doc.source_type == "policy"}) < min_policy_docs:
                selected_policy_fingerprints = {policy_fingerprint(doc) for doc in selected if doc.source_type == "policy"}
                diverse_policy_candidate = next(
                    (
                        doc
                        for doc in ranked
                        if doc.source_type == "policy"
                        and (doc.source_type, doc.doc_id) not in selected_keys
                        and policy_fingerprint(doc) not in selected_policy_fingerprints
                    ),
                    None,
                )
                if diverse_policy_candidate is None:
                    break
                seen_policy_fingerprints: set[str] = set()
                duplicate_policy_index = None
                for index, doc in enumerate(selected):
                    if doc.source_type != "policy":
                        continue
                    fingerprint = policy_fingerprint(doc)
                    if fingerprint in seen_policy_fingerprints:
                        duplicate_policy_index = index
                        break
                    seen_policy_fingerprints.add(fingerprint)
                if duplicate_policy_index is not None:
                    replace_at(duplicate_policy_index, diverse_policy_candidate)
                else:
                    add_or_replace(diverse_policy_candidate)

            while sum(1 for doc in selected if doc.source_type == "policy") < min_policy_docs:
                policy_candidate = next(
                    (
                        doc
                        for doc in ranked
                        if doc.source_type == "policy" and (doc.source_type, doc.doc_id) not in selected_keys
                    ),
                    None,
                )
                if policy_candidate is None:
                    break
                add_or_replace(policy_candidate)

        if ticket is not None:
            min_kb_docs = min(3, self.top_k)

            def add_kb_candidate(candidate: RetrievedDoc) -> None:
                candidate_key = (candidate.source_type, candidate.doc_id)
                if candidate_key in selected_keys:
                    return
                if len(selected) < self.top_k:
                    selected.append(candidate)
                    selected_keys.add(candidate_key)
                    return

                source_counts = Counter(doc.source_type for doc in selected)
                replacement_index = None
                for index in range(len(selected) - 1, -1, -1):
                    if selected[index].source_type == "history" and source_counts["history"] > 1:
                        replacement_index = index
                        break
                if replacement_index is None:
                    for index in range(len(selected) - 1, -1, -1):
                        source = selected[index].source_type
                        if source not in {"policy", "kb"} and source_counts[source] > 1:
                            replacement_index = index
                            break
                if replacement_index is None:
                    for index in range(len(selected) - 1, -1, -1):
                        if selected[index].source_type == "policy" and source_counts["policy"] > 2:
                            replacement_index = index
                            break
                if replacement_index is None:
                    replacement_index = len(selected) - 1
                replace_at(replacement_index, candidate)

            while sum(1 for doc in selected if doc.source_type == "kb") < min_kb_docs:
                kb_candidate = next(
                    (
                        doc
                        for doc in ranked
                        if doc.source_type == "kb" and (doc.source_type, doc.doc_id) not in selected_keys
                    ),
                    None,
                )
                if kb_candidate is None:
                    break
                add_kb_candidate(kb_candidate)

        return sorted(selected, key=lambda item: item.rerank_score or item.score, reverse=True)[: self.top_k]
