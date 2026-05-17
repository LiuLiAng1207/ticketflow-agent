from __future__ import annotations


def _first_ticket(runner, *, category: str, with_order: bool | None = None):
    for ticket in runner.list_open_tickets(limit=200):
        if ticket.expected_category != category:
            continue
        has_order = bool(ticket.linked_order_id)
        if with_order is None or has_order is with_order:
            return ticket
    raise AssertionError(f"No ticket found for category={category}, with_order={with_order}")


def test_payment_screenshot_is_parsed_into_attachment_evidence_and_context(runner):
    ticket = _first_ticket(runner, category="billing_refund", with_order=True)
    attachment = runner.repository.create_ticket_attachment(
        ticket_id=ticket.ticket_id,
        filename="payment_success.png",
        file_type="image",
        source_dataset="synthetic_payment_screenshot",
        ocr_text=f"支付成功 订单号 {ticket.linked_order_id} 金额 299.00 元",
        visual_summary="用户上传的支付截图显示订单已支付成功。",
    )

    result = runner.run_ticket(ticket.ticket_id, thread_id="multimodal-payment")

    evidence = result.state["attachment_evidence"]
    assert evidence
    assert evidence[0].attachment_id == attachment.attachment_id
    assert evidence[0].evidence_type == "payment_screenshot"
    assert evidence[0].entities["order_id"] == ticket.linked_order_id
    assert any(doc.source_type == "attachment" and doc.doc_id == evidence[0].evidence_id for doc in result.state["retrieved_docs"])
    assert any(item["step"] == "parse_attachments" for item in result.state["trace"])
    assert any(event.event_type == "attachments_parsed" for event in result.state["audit_log"])


def test_attachment_payment_claim_does_not_replace_missing_order_source(runner):
    ticket = _first_ticket(runner, category="billing_refund", with_order=False)
    runner.repository.create_ticket_attachment(
        ticket_id=ticket.ticket_id,
        filename="payment_success_without_verified_order.png",
        file_type="image",
        source_dataset="synthetic_payment_screenshot",
        ocr_text="支付成功 订单号 ORD-9999 金额 299.00 元",
        visual_summary="截图疑似显示支付成功，但订单系统尚未核验。",
    )

    result = runner.run_ticket(ticket.ticket_id, thread_id="multimodal-missing-order")

    assert result.state["sufficiency_result"].sufficient is False
    assert "order" in result.state["sufficiency_result"].missing_sources
    assert result.state["proposed_action"].action_type == "request_info"
    assert any(doc.source_type == "attachment" for doc in result.state["retrieved_docs"])
