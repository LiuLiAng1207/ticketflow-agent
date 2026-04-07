from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import TicketFlowRepository, _query_terms
from .models import Citation, RetrievedDoc, RetrievalStats, TicketRecord

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None


def _normalize_text(text: str) -> str:
    return " ".join(_query_terms(text))


def _category_keywords(ticket: TicketRecord) -> list[str]:
    mapping = {
        "billing_refund": ["退款", "账单", "refund", "billing"],
        "delivery_issue": ["物流", "发货", "delivery", "shipping"],
        "technical_issue": ["故障", "生产", "technical", "incident"],
        "account_access": ["登录", "账号", "account", "access"],
        "general_inquiry": ["咨询", "说明", "general", "faq"],
    }
    category = ticket.expected_category or "general_inquiry"
    return mapping.get(category, mapping["general_inquiry"])


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    embedding_fn: HashEmbeddingFunction | None = None
    collection: Any | None = None

    def __post_init__(self) -> None:
        self.rag_db_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = HashEmbeddingFunction()
        if self.enabled and chromadb is not None:
            client = chromadb.PersistentClient(path=str(self.rag_db_dir))
            self.collection = client.get_or_create_collection(
                name="ticketflow_rag",
                embedding_function=self.embedding_fn,
                metadata={"source": "ticketflow", "embed_model": self.embed_model},
            )
            self._sync_collection()

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
            documents.append(
                {
                    "id": f"history:{row['event_id']}",
                    "document": f"{row['message']}\n{row['agent_name']}",
                    "metadata": {
                        "source_type": "history",
                        "doc_id": row["event_id"],
                        "ticket_id": row["ticket_id"],
                        "title": f"历史处理记录 {row['event_id']}",
                        "agent_name": row["agent_name"],
                        "snippet": row["message"],
                    },
                }
            )
        return documents

    def _sync_collection(self) -> None:
        if self.collection is None:
            return
        documents = self._corpus_documents()
        if not documents:
            return
        ids = [doc["id"] for doc in documents]
        payloads = [doc["document"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        self.collection.upsert(ids=ids, documents=payloads, metadatas=metadatas)

    def retrieve(self, ticket: TicketRecord) -> RagSearchResult:
        query = f"{ticket.title}\n{ticket.body}\n{ticket.product}\n{' '.join(_category_keywords(ticket))}"
        lexical_docs = self._lexical_search(ticket, query)
        merged_docs: dict[tuple[str, str], RetrievedDoc] = {
            (doc.source_type, doc.doc_id): doc for doc in lexical_docs
        }

        vector_hits = 0
        if self.collection is not None:
            vector_docs = self._vector_search(query, self.top_k * 2)
            vector_hits = len(vector_docs)
            for doc in vector_docs:
                key = (doc.source_type, doc.doc_id)
                if key in merged_docs:
                    merged_docs[key].score = max(merged_docs[key].score, doc.score) + 0.15
                else:
                    merged_docs[key] = doc

        final_docs = self._select_balanced_docs(list(merged_docs.values()))
        source_counts = Counter(doc.source_type for doc in final_docs)
        stats = RetrievalStats(
            lexical_hits=len(lexical_docs),
            vector_hits=vector_hits,
            final_hits=len(final_docs),
            used_vector=vector_hits > 0,
            fallback_to_keywords=self.collection is None,
            source_counts=dict(source_counts),
            retrieved_doc_ids=[f"{doc.source_type}:{doc.doc_id}" for doc in final_docs],
        )
        return RagSearchResult(docs=final_docs, stats=stats)

    def _lexical_search(self, ticket: TicketRecord, query: str) -> list[RetrievedDoc]:
        docs: list[RetrievedDoc] = []
        seen: set[tuple[str, str]] = set()
        for row in self.repository.search_kb(query, ticket.product, limit=self.top_k):
            doc = RetrievedDoc(
                doc_id=str(row["doc_id"]),
                source_type="kb",
                title=str(row["title"]),
                snippet=str(row["body"]),
                score=float(row.get("score", 0.0)),
                metadata={"product": row.get("product"), "category": row.get("category"), "retrieval": "keyword"},
                citations=[Citation(source_path=f"kb:{row['doc_id']}", note="知识库条目")],
            )
            if (doc.source_type, doc.doc_id) not in seen:
                docs.append(doc)
                seen.add((doc.source_type, doc.doc_id))

        for row in self.repository.search_policy_text(query, limit=max(2, self.top_k // 2)):
            doc = RetrievedDoc(
                doc_id=str(row["policy_id"]),
                source_type="policy",
                title=str(row["title"]),
                snippet=str(row["body"]),
                score=float(row.get("score", 0.0)),
                metadata={
                    "action_type": row.get("action_type"),
                    "approval_required": bool(row.get("approval_required")),
                    "priority_hint": row.get("priority_hint"),
                    "retrieval": "keyword",
                },
                citations=[Citation(source_path=f"policy:{row['policy_id']}", note="策略规则")],
            )
            if (doc.source_type, doc.doc_id) not in seen:
                docs.append(doc)
                seen.add((doc.source_type, doc.doc_id))

        for row in self.repository.search_related_history(query, limit=max(2, self.top_k // 2)):
            doc = RetrievedDoc(
                doc_id=str(row["event_id"]),
                source_type="history",
                title=f"历史处理记录 {row['event_id']}",
                snippet=str(row["message"]),
                score=float(row.get("score", 0.0)),
                metadata={
                    "ticket_id": row.get("ticket_id"),
                    "agent_name": row.get("agent_name"),
                    "retrieval": "keyword",
                },
                citations=[Citation(source_path=f"history:{row['event_id']}", note="历史工单经验")],
            )
            if (doc.source_type, doc.doc_id) not in seen:
                docs.append(doc)
                seen.add((doc.source_type, doc.doc_id))
        return docs

    def _vector_search(self, query: str, limit: int) -> list[RetrievedDoc]:
        if self.collection is None:
            return []
        query_embedding = self.embedding_fn.embed_query(_normalize_text(query)) if self.embedding_fn else []
        raw = self.collection.query(query_embeddings=[query_embedding], n_results=limit)
        ids = raw.get("ids", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else [0.0] * len(ids)
        docs: list[RetrievedDoc] = []
        for idx, metadata in enumerate(metadatas):
            if not metadata:
                continue
            source_type = metadata["source_type"]
            doc_id = metadata["doc_id"]
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)
            note = {
                "kb": "知识库条目",
                "policy": "策略规则",
                "history": "历史工单经验",
            }.get(source_type, "检索证据")
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
                    citations=[Citation(source_path=f"kb:{row['doc_id']}", note="知识库条目")],
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
            "embed_model": self.embed_model,
            "backend": "chromadb" if self.collection is not None else "keyword_only",
            "doc_count": len(self._corpus_documents()),
        }
        (self.rag_db_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _select_balanced_docs(self, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        by_source: dict[str, list[RetrievedDoc]] = {}
        for doc in sorted(docs, key=lambda item: item.score, reverse=True):
            by_source.setdefault(doc.source_type, []).append(doc)

        selected: list[RetrievedDoc] = []
        seen: set[tuple[str, str]] = set()
        for source in ("policy", "kb", "history"):
            if not by_source.get(source):
                continue
            doc = by_source[source][0]
            key = (doc.source_type, doc.doc_id)
            if key not in seen:
                selected.append(doc)
                seen.add(key)

        for doc in sorted(docs, key=lambda item: item.score, reverse=True):
            key = (doc.source_type, doc.doc_id)
            if key in seen:
                continue
            selected.append(doc)
            seen.add(key)
            if len(selected) >= self.top_k:
                break
        return selected[: self.top_k]
