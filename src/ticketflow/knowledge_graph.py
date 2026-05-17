from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from .models import TicketRecord
from .repository import RepositoryProtocol
from .service_settings import ServiceSettings


CONTROLLED_NODE_LABELS = {
    "Ticket",
    "Customer",
    "Order",
    "Product",
    "Policy",
    "KBArticle",
    "HistoryCase",
    "AttachmentEvidence",
    "Action",
    "ToolCall",
    "Approval",
    "EmailDelivery",
    "WorkflowTask",
}
CONTROLLED_EDGE_TYPES = {
    "CUSTOMER_OPENED_TICKET",
    "TICKET_LINKED_ORDER",
    "TICKET_FOR_PRODUCT",
    "TICKET_HAS_EVIDENCE",
    "EVIDENCE_SUPPORTS_ACTION",
    "ACTION_REQUIRES_APPROVAL",
    "ACTION_EXECUTED_TOOL",
    "APPROVAL_DECIDED_ACTION",
    "TICKET_EMITTED_OUTBOX",
    "WORKFLOW_TASK_PROCESSED_TICKET",
}


class KnowledgeGraphStore(Protocol):
    def health(self) -> dict[str, Any]: ...

    def upsert_ticket_graph(self, graph: dict[str, Any]) -> dict[str, Any]: ...

    def get_ticket_graph(self, ticket_id: str) -> dict[str, Any]: ...

    def search(self, query: str, limit: int = 20) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _node(node_id: str, label: str, **properties: Any) -> dict[str, Any]:
    safe_label = label if label in CONTROLLED_NODE_LABELS else "AttachmentEvidence"
    return {"id": node_id, "label": safe_label, "properties": _json_safe(properties)}


def _edge(source: str, edge_type: str, target: str, **properties: Any) -> dict[str, Any]:
    safe_type = edge_type if edge_type in CONTROLLED_EDGE_TYPES else "TICKET_HAS_EVIDENCE"
    return {"source": source, "type": safe_type, "target": target, "properties": _json_safe(properties)}


def _add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    existing = nodes.get(node["id"])
    if existing is None:
        nodes[node["id"]] = node
        return
    existing["properties"].update(node.get("properties") or {})


def _add_edge(edges: dict[tuple[str, str, str], dict[str, Any]], edge: dict[str, Any]) -> None:
    key = (edge["source"], edge["type"], edge["target"])
    existing = edges.get(key)
    if existing is None:
        edges[key] = edge
        return
    existing["properties"].update(edge.get("properties") or {})


def _doc_label(source_type: str) -> str:
    return {
        "policy": "Policy",
        "kb": "KBArticle",
        "history": "HistoryCase",
        "attachment": "AttachmentEvidence",
        "order": "Order",
        "customer": "Customer",
    }.get(source_type, "AttachmentEvidence")


def _doc_node_id(doc: dict[str, Any]) -> str:
    doc_id = str(doc.get("doc_id") or doc.get("id") or "")
    source_type = str(doc.get("source_type") or "evidence")
    if ":" in doc_id:
        return doc_id
    return f"{source_type}:{doc_id}"


def _extract_state_from_task(task: dict[str, Any]) -> dict[str, Any]:
    result = task.get("result")
    if isinstance(result, dict):
        state = result.get("state")
        if isinstance(state, dict):
            return state
    return {}


def _list_for_ticket(items: list[dict[str, Any]], ticket_id: str) -> list[dict[str, Any]]:
    return [item for item in items if str(item.get("ticket_id")) == ticket_id]


def build_ticket_graph_from_repository(
    repository: RepositoryProtocol,
    ticket_id: str,
    *,
    task_id: str | None = None,
) -> dict[str, Any]:
    ticket = repository.get_ticket(ticket_id)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    ticket_node_id = f"ticket:{ticket.ticket_id}"
    _add_node(nodes, _node(ticket_node_id, "Ticket", **ticket.model_dump(mode="json")))

    if ticket.customer_id:
        customer_node_id = f"customer:{ticket.customer_id}"
        customer_payload = {"customer_id": ticket.customer_id, "customer_tier": ticket.customer_tier}
        if hasattr(repository, "get_customer_profile"):
            profile = repository.get_customer_profile(ticket.customer_id)  # type: ignore[attr-defined]
            if profile is not None:
                customer_payload.update(profile.model_dump(mode="json"))
        _add_node(nodes, _node(customer_node_id, "Customer", **customer_payload))
        _add_edge(edges, _edge(customer_node_id, "CUSTOMER_OPENED_TICKET", ticket_node_id))

    if ticket.product:
        product_node_id = f"product:{ticket.product}"
        _add_node(nodes, _node(product_node_id, "Product", name=ticket.product))
        _add_edge(edges, _edge(ticket_node_id, "TICKET_FOR_PRODUCT", product_node_id))

    if ticket.linked_order_id:
        order_node_id = f"order:{ticket.linked_order_id}"
        order_payload: dict[str, Any] = {"order_id": ticket.linked_order_id}
        if hasattr(repository, "get_order_status"):
            order = repository.get_order_status(ticket.linked_order_id)  # type: ignore[attr-defined]
            if order is not None:
                order_payload.update(order.model_dump(mode="json"))
        _add_node(nodes, _node(order_node_id, "Order", **order_payload))
        _add_edge(edges, _edge(ticket_node_id, "TICKET_LINKED_ORDER", order_node_id))

    if hasattr(repository, "list_attachment_evidence"):
        for evidence in repository.list_attachment_evidence(ticket.ticket_id):  # type: ignore[attr-defined]
            evidence_id = f"attachment:{evidence.evidence_id}"
            _add_node(nodes, _node(evidence_id, "AttachmentEvidence", **evidence.model_dump(mode="json")))
            _add_edge(edges, _edge(ticket_node_id, "TICKET_HAS_EVIDENCE", evidence_id))

    workflow_tasks = []
    if hasattr(repository, "list_workflow_tasks"):
        workflow_tasks = _list_for_ticket(repository.list_workflow_tasks(limit=500), ticket.ticket_id)  # type: ignore[attr-defined]
        if task_id is not None:
            workflow_tasks = [task for task in workflow_tasks if str(task.get("task_id")) == task_id]

    action_node_id: str | None = None
    for task in workflow_tasks:
        workflow_node_id = f"workflow_task:{task['task_id']}"
        _add_node(nodes, _node(workflow_node_id, "WorkflowTask", **task))
        _add_edge(edges, _edge(workflow_node_id, "WORKFLOW_TASK_PROCESSED_TICKET", ticket_node_id))

        state = _extract_state_from_task(task)
        proposed_action = state.get("proposed_action")
        if isinstance(proposed_action, dict):
            action_type = str(proposed_action.get("action_type") or "unknown")
            action_node_id = f"action:{ticket.ticket_id}:{action_type}"
            _add_node(nodes, _node(action_node_id, "Action", **proposed_action))
            _add_edge(edges, _edge(ticket_node_id, "ACTION_EXECUTED_TOOL", action_node_id))

            tool_name = proposed_action.get("suggested_tool")
            if tool_name:
                tool_node_id = f"tool_call:{ticket.ticket_id}:{tool_name}"
                _add_node(nodes, _node(tool_node_id, "ToolCall", tool_name=tool_name, tool_args=proposed_action.get("tool_args") or {}))
                _add_edge(edges, _edge(action_node_id, "ACTION_EXECUTED_TOOL", tool_node_id))

        for doc in state.get("retrieved_docs") or []:
            if not isinstance(doc, dict):
                continue
            doc_node_id = _doc_node_id(doc)
            _add_node(nodes, _node(doc_node_id, _doc_label(str(doc.get("source_type") or "")), **doc))
            _add_edge(edges, _edge(ticket_node_id, "TICKET_HAS_EVIDENCE", doc_node_id))
            if action_node_id:
                _add_edge(edges, _edge(doc_node_id, "EVIDENCE_SUPPORTS_ACTION", action_node_id))

    if hasattr(repository, "list_audit_log"):
        for event in repository.list_audit_log(ticket_id=ticket.ticket_id, limit=500):
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            for doc_id in payload.get("supporting_doc_ids", []) if isinstance(payload, dict) else []:
                if not isinstance(doc_id, str):
                    continue
                source_type = doc_id.split(":", 1)[0] if ":" in doc_id else "evidence"
                _add_node(nodes, _node(doc_id, _doc_label(source_type), doc_id=doc_id, source_type=source_type))
                _add_edge(edges, _edge(ticket_node_id, "TICKET_HAS_EVIDENCE", doc_id))
                if action_node_id:
                    _add_edge(edges, _edge(doc_id, "EVIDENCE_SUPPORTS_ACTION", action_node_id))

    approvals = []
    if hasattr(repository, "list_approval_requests"):
        approvals = _list_for_ticket(repository.list_approval_requests(limit=500), ticket.ticket_id)  # type: ignore[attr-defined]
    for approval in approvals:
        approval_node_id = f"approval:{approval['approval_id']}"
        _add_node(nodes, _node(approval_node_id, "Approval", **approval))
        if action_node_id:
            _add_edge(edges, _edge(action_node_id, "ACTION_REQUIRES_APPROVAL", approval_node_id))
            _add_edge(edges, _edge(approval_node_id, "APPROVAL_DECIDED_ACTION", action_node_id, status=approval.get("status")))

    outbox_events = []
    if hasattr(repository, "list_outbox_events"):
        outbox_events = _list_for_ticket(repository.list_outbox_events(limit=500), ticket.ticket_id)  # type: ignore[attr-defined]
    for outbox in outbox_events:
        outbox_node_id = f"outbox:{outbox['event_id']}"
        _add_node(nodes, _node(outbox_node_id, "EmailDelivery", **outbox))
        _add_edge(edges, _edge(ticket_node_id, "TICKET_EMITTED_OUTBOX", outbox_node_id))

    graph = {
        "enabled": True,
        "ticket_id": ticket.ticket_id,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }
    graph["node_count"] = len(graph["nodes"])
    graph["edge_count"] = len(graph["edges"])
    return graph


@dataclass(slots=True)
class DisabledKnowledgeGraphStore:
    reason: str = "knowledge graph backend is disabled"

    def health(self) -> dict[str, Any]:
        return {"status": "disabled", "backend": "disabled", "reason": self.reason}

    def upsert_ticket_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        return {"status": "disabled", "backend": "disabled", "reason": self.reason, "ticket_id": graph.get("ticket_id")}

    def get_ticket_graph(self, ticket_id: str) -> dict[str, Any]:
        return {"enabled": False, "ticket_id": ticket_id, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

    def search(self, query: str, limit: int = 20) -> dict[str, Any]:
        return {"enabled": False, "query": query, "results": [], "count": 0, "limit": limit}

    def close(self) -> None:
        return None


@dataclass(slots=True)
class InMemoryKnowledgeGraphStore:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    ticket_index: dict[str, set[str]] = field(default_factory=dict)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "memory", "node_count": len(self.nodes), "edge_count": len(self.edges)}

    def upsert_ticket_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        ticket_id = str(graph.get("ticket_id"))
        created_nodes = 0
        created_edges = 0
        ticket_nodes = self.ticket_index.setdefault(ticket_id, set())
        for node in graph.get("nodes", []):
            node_id = str(node["id"])
            if node_id not in self.nodes:
                created_nodes += 1
            self.nodes[node_id] = node
            ticket_nodes.add(node_id)
        for edge in graph.get("edges", []):
            key = (str(edge["source"]), str(edge["type"]), str(edge["target"]))
            if key not in self.edges:
                created_edges += 1
            self.edges[key] = edge
            ticket_nodes.update({key[0], key[2]})
        return {
            "status": "ok",
            "backend": "memory",
            "ticket_id": ticket_id,
            "created_nodes": created_nodes,
            "created_edges": created_edges,
            "node_count": len(ticket_nodes),
            "edge_count": sum(1 for key in self.edges if key[0] in ticket_nodes or key[2] in ticket_nodes),
        }

    def get_ticket_graph(self, ticket_id: str) -> dict[str, Any]:
        ticket_nodes = self.ticket_index.get(ticket_id, set())
        edges = [edge for key, edge in self.edges.items() if key[0] in ticket_nodes or key[2] in ticket_nodes]
        nodes = [self.nodes[node_id] for node_id in sorted(ticket_nodes) if node_id in self.nodes]
        return {
            "enabled": True,
            "backend": "memory",
            "ticket_id": ticket_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def search(self, query: str, limit: int = 20) -> dict[str, Any]:
        needle = query.lower()
        results = []
        for node in self.nodes.values():
            haystack = " ".join([str(node.get("id", "")), str(node.get("label", "")), str(node.get("properties", ""))]).lower()
            if needle in haystack:
                results.append(node)
            if len(results) >= limit:
                break
        return {"enabled": True, "backend": "memory", "query": query, "results": results, "count": len(results), "limit": limit}

    def close(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.ticket_index.clear()


class Neo4jKnowledgeGraphStore:
    def __init__(self, uri: str, user: str | None, password: str | None):
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime install.
            raise RuntimeError("Neo4j driver is not installed. Install `neo4j>=5.25.0`.") from exc
        if not password:
            raise ValueError("NEO4J_PASSWORD is required when KG_BACKEND=neo4j")
        self._driver = GraphDatabase.driver(uri, auth=(user or "neo4j", password))

    def health(self) -> dict[str, Any]:
        try:
            self._driver.verify_connectivity()
            return {"status": "ok", "backend": "neo4j"}
        except Exception as exc:  # noqa: BLE001 - KG must not break the core platform.
            return {"status": "unavailable", "backend": "neo4j", "error": str(exc)}

    def upsert_ticket_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        ticket_id = str(graph.get("ticket_id"))
        try:
            with self._driver.session() as session:
                for node in graph.get("nodes", []):
                    label = _safe_cypher_symbol(str(node["label"]), CONTROLLED_NODE_LABELS)
                    session.run(
                        f"MERGE (n:{label} {{id: $id}}) SET n += $properties, n.ticket_id = $ticket_id",
                        id=str(node["id"]),
                        properties=node.get("properties") or {},
                        ticket_id=ticket_id,
                    )
                for edge in graph.get("edges", []):
                    edge_type = _safe_cypher_symbol(str(edge["type"]), CONTROLLED_EDGE_TYPES)
                    session.run(
                        f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[rel:{edge_type}]->(target)
                        SET rel += $properties, rel.ticket_id = $ticket_id
                        """,
                        source_id=str(edge["source"]),
                        target_id=str(edge["target"]),
                        properties=edge.get("properties") or {},
                        ticket_id=ticket_id,
                    )
        except Exception as exc:  # noqa: BLE001 - KG is an explanation layer, not a core workflow dependency.
            return {
                "status": "unavailable",
                "backend": "neo4j",
                "ticket_id": ticket_id,
                "error": str(exc),
                "created_nodes": 0,
                "created_edges": 0,
                "node_count": len(graph.get("nodes", [])),
                "edge_count": len(graph.get("edges", [])),
            }
        return {
            "status": "ok",
            "backend": "neo4j",
            "ticket_id": ticket_id,
            "created_nodes": 0,
            "created_edges": 0,
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
        }

    def get_ticket_graph(self, ticket_id: str) -> dict[str, Any]:
        try:
            with self._driver.session() as session:
                nodes = [
                    {"id": row["id"], "label": row["label"], "properties": row["properties"]}
                    for row in session.run(
                        """
                        MATCH (n {ticket_id: $ticket_id})
                        RETURN n.id AS id, labels(n)[0] AS label, properties(n) AS properties
                        ORDER BY id
                        """,
                        ticket_id=ticket_id,
                    )
                ]
                edges = [
                    {"source": row["source"], "type": row["type"], "target": row["target"], "properties": row["properties"]}
                    for row in session.run(
                        """
                        MATCH (source)-[rel {ticket_id: $ticket_id}]->(target)
                        RETURN source.id AS source, type(rel) AS type, target.id AS target, properties(rel) AS properties
                        ORDER BY source, type, target
                        """,
                        ticket_id=ticket_id,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            return {
                "enabled": False,
                "backend": "neo4j",
                "ticket_id": ticket_id,
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "error": str(exc),
            }
        return {
            "enabled": True,
            "backend": "neo4j",
            "ticket_id": ticket_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def search(self, query: str, limit: int = 20) -> dict[str, Any]:
        pattern = f"(?i).*{re.escape(query)}.*"
        try:
            with self._driver.session() as session:
                results = [
                    {"id": row["id"], "label": row["label"], "properties": row["properties"]}
                    for row in session.run(
                        """
                        MATCH (n)
                        WHERE n.id =~ $pattern OR any(label IN labels(n) WHERE label =~ $pattern)
                        RETURN n.id AS id, labels(n)[0] AS label, properties(n) AS properties
                        LIMIT $limit
                        """,
                        pattern=pattern,
                        limit=limit,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            return {"enabled": False, "backend": "neo4j", "query": query, "results": [], "count": 0, "limit": limit, "error": str(exc)}
        return {"enabled": True, "backend": "neo4j", "query": query, "results": results, "count": len(results), "limit": limit}

    def close(self) -> None:
        self._driver.close()


def _safe_cypher_symbol(value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported graph symbol: {value}")
    return value


def create_knowledge_graph_store(settings: ServiceSettings) -> KnowledgeGraphStore:
    if settings.kg_backend == "memory":
        return InMemoryKnowledgeGraphStore()
    if settings.kg_backend == "neo4j":
        if not settings.neo4j_uri:
            return DisabledKnowledgeGraphStore(reason="NEO4J_URI is not configured")
        return Neo4jKnowledgeGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    return DisabledKnowledgeGraphStore(reason="KG_BACKEND=disabled")
