from __future__ import annotations

from ticketflow.knowledge_graph import (
    DisabledKnowledgeGraphStore,
    InMemoryKnowledgeGraphStore,
    build_ticket_graph_from_repository,
)


def test_disabled_knowledge_graph_store_is_safe_fallback():
    store = DisabledKnowledgeGraphStore(reason="neo4j not configured")

    health = store.health()
    graph = store.get_ticket_graph("TCK-0001")
    search = store.search("refund")
    rebuild = store.upsert_ticket_graph(graph)

    assert health["status"] == "disabled"
    assert health["reason"] == "neo4j not configured"
    assert graph["enabled"] is False
    assert graph["nodes"] == []
    assert graph["edges"] == []
    assert search["results"] == []
    assert rebuild["status"] == "disabled"


def test_ticket_graph_builder_collects_business_objects_and_is_idempotent(runner):
    ticket = next(
        item
        for item in runner.list_open_tickets(limit=500)
        if item.linked_order_id and item.expected_category == "billing_refund"
    )
    runner.repository.save_audit_log(
        ticket.ticket_id,
        actor="test",
        event_type="retrieval_completed",
        detail="test evidence",
        payload={"supporting_doc_ids": ["policy:POL-001-01", "history:HIS-0001"]},
    )
    task = runner.repository.create_workflow_task(ticket_id=ticket.ticket_id, mode="async")
    runner.repository.complete_workflow_task(
        str(task["task_id"]),
        result={
            "state": {
                "thread_id": "thread-kg-test",
                "ticket": ticket.model_dump(mode="json"),
                "retrieved_docs": [
                    {
                        "doc_id": "policy:POL-001-01",
                        "source_type": "policy",
                        "title": "退款策略",
                        "snippet": "企业客户退款需要订单和策略证据。",
                        "score": 0.91,
                        "metadata": {"policy_id": "POL-001-01"},
                        "citations": [],
                    },
                    {
                        "doc_id": "history:HIS-0001",
                        "source_type": "history",
                        "title": "相似退款案例",
                        "snippet": "相似会员不可用退款案例。",
                        "score": 0.86,
                        "metadata": {"history_id": "HIS-0001"},
                        "citations": [],
                    },
                ],
                "proposed_action": {
                    "action_type": "refund",
                    "target_status": "pending_finance",
                    "rationale": "证据充分，进入退款审批。",
                    "requires_approval": True,
                    "suggested_tool": "issue_refund_request",
                    "tool_args": {"ticket_id": ticket.ticket_id, "order_id": ticket.linked_order_id},
                    "confidence": 0.9,
                    "route_family": "refund_candidate",
                    "sufficiency_passed": True,
                    "supporting_doc_ids": ["policy:POL-001-01", "history:HIS-0001"],
                    "missing_sources": [],
                    "decision_source": "governance",
                },
            }
        },
    )
    approval = runner.repository.create_approval_request(
        ticket_id=ticket.ticket_id,
        thread_id="thread-kg-test",
        tool_name="issue_refund_request",
        tool_args={"ticket_id": ticket.ticket_id, "order_id": ticket.linked_order_id},
        payload={"workflow_task_id": task["task_id"], "route_family": "refund_candidate"},
        requested_by="kg-test",
    )
    outbox = runner.repository.create_outbox_event(
        ticket_id=ticket.ticket_id,
        operation_type="incident_email",
        business_key=f"incident:{ticket.ticket_id}",
        payload={"subject": "测试投递"},
    )

    graph = build_ticket_graph_from_repository(runner.repository, ticket.ticket_id)
    store = InMemoryKnowledgeGraphStore()
    first = store.upsert_ticket_graph(graph)
    second = store.upsert_ticket_graph(graph)
    loaded = store.get_ticket_graph(ticket.ticket_id)

    node_ids = {node["id"] for node in graph["nodes"]}
    edge_keys = {(edge["source"], edge["type"], edge["target"]) for edge in graph["edges"]}

    assert f"ticket:{ticket.ticket_id}" in node_ids
    assert f"customer:{ticket.customer_id}" in node_ids
    assert f"order:{ticket.linked_order_id}" in node_ids
    assert "policy:POL-001-01" in node_ids
    assert "history:HIS-0001" in node_ids
    assert f"workflow_task:{task['task_id']}" in node_ids
    assert f"approval:{approval['approval_id']}" in node_ids
    assert f"outbox:{outbox['event_id']}" in node_ids
    assert (f"ticket:{ticket.ticket_id}", "TICKET_LINKED_ORDER", f"order:{ticket.linked_order_id}") in edge_keys
    assert (f"policy:POL-001-01", "EVIDENCE_SUPPORTS_ACTION", f"action:{ticket.ticket_id}:refund") in edge_keys
    assert first["created_nodes"] >= 1
    assert second["created_nodes"] == 0
    assert second["created_edges"] == 0
    assert loaded["ticket_id"] == ticket.ticket_id
    assert len(loaded["nodes"]) == len(graph["nodes"])
