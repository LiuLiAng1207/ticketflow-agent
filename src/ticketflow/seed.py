from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


PRODUCTS = ["灯塔路由器", "云桌面协作", "视觉巡检仪", "安全金库", "运维控制台"]
CHANNELS = ["email", "chat", "web"]
TIERS = ["standard", "premium", "enterprise"]
REGIONS = ["华东", "华南", "华北", "西南"]
CATEGORY_SEQUENCE = [
    "billing_refund",
    "delivery_issue",
    "technical_issue",
    "account_access",
    "general_inquiry",
]
CATEGORY_TITLES = {
    "billing_refund": "退款与账单",
    "delivery_issue": "订单履约与物流",
    "technical_issue": "技术故障",
    "account_access": "账号访问",
    "general_inquiry": "产品咨询",
}
PUBLIC_CORPUS_FILENAME = "public_ticket_corpus.csv"
PUBLIC_DATASET_NAME = "Tobi-Bueck/customer-support-tickets"


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_public_corpus_path(seed_dir: Path) -> Path:
    preferred = seed_dir.parent / "public" / PUBLIC_CORPUS_FILENAME
    bundled = Path(__file__).resolve().parents[2] / "data" / "public" / PUBLIC_CORPUS_FILENAME
    if preferred.exists():
        return preferred
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"未找到公开工单语料缓存文件 {PUBLIC_CORPUS_FILENAME}。"
        "请确认 data/public 目录存在。"
    )


def _load_public_ticket_corpus(seed_dir: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    path = _resolve_public_corpus_path(seed_dir)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["mapped_category"]].append(row)
    missing = [category for category in CATEGORY_SEQUENCE if not grouped.get(category)]
    if missing:
        raise ValueError(f"公开工单语料缺少以下类别：{', '.join(missing)}")
    return grouped


def _public_excerpt(text: str, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _public_kb_rows(public_corpus: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    kb_rows: list[dict[str, object]] = []
    article_idx = 1
    for category in CATEGORY_SEQUENCE:
        rows = public_corpus[category]
        for repetition in range(24):
            source_row = rows[repetition % len(rows)]
            product = PRODUCTS[repetition % len(PRODUCTS)]
            queue = source_row.get("source_queue", "").strip() or "Support"
            body_excerpt = _public_excerpt(source_row.get("source_body", ""))
            kb_rows.append(
                {
                    "doc_id": f"KB-{article_idx:03d}",
                    "title": f"{CATEGORY_TITLES[category]}案例摘要 {repetition + 1}",
                    "body": (
                        f"公开工单主题：{source_row.get('source_subject', '')}。"
                        f"公开工单摘录：{body_excerpt}。"
                        f"来源队列：{queue}。"
                        f"建议结合产品“{product}”与当前工单上下文继续核验。"
                    ),
                    "tags": f"{category},{product},support,public_corpus",
                    "product": product,
                    "category": category,
                }
            )
            article_idx += 1
    return kb_rows


def _kb_templates() -> dict[str, list[str]]:
    return {
        "billing_refund": [
            "已签收订单在 30 天内且订单号可核验时，可以进入退款审核流程。",
            "若客户没有提供订单号，客服必须先索要订单编号或付款截图，不能直接承诺退款。",
            "企业版客户可以申请部分退款或账单抵扣，但最终退款动作必须经过主管审批。",
        ],
        "delivery_issue": [
            "物流延迟类工单需要同步最新订单状态、预计送达时间和当前处理人。",
            "订单连续 5 天以上处于待发货或运输停滞状态时，应创建履约或物流升级单。",
            "若客户反馈包裹丢失，应先核对收货信息，再决定是否升级给承运商。",
        ],
        "technical_issue": [
            "企业客户出现生产阻塞或 Sev1 故障时，15 分钟内必须通知值班专家。",
            "技术故障排查时要优先收集报错信息、浏览器版本、影响范围和是否有临时绕过方案。",
            "如果多个用户同时受影响，应优先考虑系统性故障并启动升级流程。",
        ],
        "account_access": [
            "账号登录失败时应先引导客户检查重置链接、验证码和 MFA 配置。",
            "未收到客户确认前，不得主动关闭账号访问类工单。",
            "若 MFA 在多人侧同时失效，应按技术故障提高优先级。",
        ],
        "general_inquiry": [
            "普通产品咨询可以引用产品文档、上手指南和常见问题说明。",
            "涉及路线图或功能上线时间时，应避免做超出政策的明确承诺。",
            "报价或商务方案问题应提供套餐概览，并将企业报价需求转给销售团队。",
        ],
    }


def _reply_template_bodies() -> dict[str, str]:
    return {
        "billing_refund": "我们已经收到你的退款诉求，正在根据订单信息核验下一步处理方案。",
        "delivery_issue": "我们正在核对订单履约状态，会尽快同步最新物流进展。",
        "technical_issue": "我们已经开始排查故障，会结合影响范围和优先级推进处理。",
        "account_access": "我们会先协助你核对账号访问问题，再给出下一步处理方案。",
        "general_inquiry": "我们已经收到你的咨询，会结合产品资料给出清晰答复。",
    }


def _issue_hint(category: str, source_subject: str, source_body: str) -> str:
    text = f"{source_subject} {source_body}".lower()

    def _has_any(phrases: list[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    if category == "billing_refund":
        if _has_any(["invoice", "billing"]):
            return "账单金额或发票信息异常"
        if _has_any(["charged twice", "overbilled", "overcharge", "charge"]):
            return "重复扣费或金额异常"
        if _has_any(["refund", "return", "exchange"]):
            return "退款与售后处理异常"
        return "退款、账单或扣费异常"
    if category == "delivery_issue":
        if _has_any(["tracking"]):
            return "物流轨迹长时间未更新"
        if _has_any(["fulfillment", "fulfilment", "dispatch"]):
            return "订单履约进度异常"
        if _has_any(["shipment", "shipping", "delivery", "arrival"]):
            return "发货与配送状态异常"
        return "发货、物流或履约进度异常"
    if category == "technical_issue":
        if _has_any(["security", "breach"]):
            return "安全告警或权限异常影响使用"
        if _has_any(["outage", "unavailable", "downtime"]):
            return "服务中断或不可用影响团队使用"
        if _has_any(["latency", "slow", "performance"]):
            return "性能波动影响正常使用"
        if _has_any(["crash", "crashed", "error", "failed", "failure", "bug"]):
            return "系统报错或故障需要排查"
        return "系统报错、安全告警或服务中断"
    if category == "account_access":
        if _has_any(["mfa", "2fa", "otp", "verification"]):
            return "多因素认证或验证码异常"
        if _has_any(["password", "credential", "unlock"]):
            return "密码重置或凭证校验失败"
        return "登录、密码或多因素认证异常"
    if _has_any(["pricing", "quote"]):
        return "报价与套餐方案咨询"
    if _has_any(["feature", "roadmap", "launch"]):
        return "功能能力与上线计划咨询"
    return "产品能力、上线计划或商务咨询"


def _ticket_title(category: str, issue_hint: str, idx: int) -> str:
    patterns = {
        "billing_refund": [
            "想确认最近订单的退款处理",
            "账单与扣费异常，需要尽快核对",
            "售后退款流程想尽快推进",
            f"{issue_hint}，需要人工处理",
        ],
        "delivery_issue": [
            "订单发货进度长时间未更新",
            "物流履约出现异常，想尽快确认",
            "包裹运输状态异常，需要协助排查",
            f"{issue_hint}，请帮忙核对",
        ],
        "technical_issue": [
            "系统异常已经影响团队使用",
            "服务报错持续出现，需要尽快排查",
            "故障影响正常业务推进",
            f"{issue_hint}，请优先处理",
        ],
        "account_access": [
            "重置密码后仍然无法登录",
            "账号验证一直失败，影响使用",
            "登录与权限访问出现异常",
            f"{issue_hint}，需要协助处理",
        ],
        "general_inquiry": [
            "想咨询产品功能与上线安排",
            "需要确认方案能力和适用方式",
            "想了解商务报价和功能边界",
            f"{issue_hint}，希望尽快回复",
        ],
    }
    options = patterns[category]
    return options[idx % len(options)]


def _ticket_body(
    *,
    category: str,
    issue_hint: str,
    product: str,
    order_id: str | None,
    missing_order: bool,
    customer_tier: str,
    idx: int,
) -> str:
    if category == "billing_refund":
        if missing_order:
            return (
                f"我们在使用“{product}”过程中遇到了{issue_hint}，现在想申请退款或补偿。"
                "我手头暂时找不到订单号，但可以补充付款截图和采购信息，请先帮我确认处理规则。"
            )
        templates = [
            f"关于订单 {order_id} 对应的“{product}”，目前遇到{issue_hint}，想确认是否满足退款条件，并尽快推进售后处理。",
            f"订单 {order_id} 购买的“{product}”最近使用体验不符合预期，同时还出现了{issue_hint}，请帮我核对退款或账单修正流程。",
            f"我们希望针对订单 {order_id} 申请退款，原因是“{product}”当前存在{issue_hint}，请结合订单状态给出下一步处理建议。",
        ]
        return templates[idx % len(templates)]
    if category == "delivery_issue":
        templates = [
            f"订单 {order_id} 对应的“{product}”出现了{issue_hint}，我们长时间没有看到新的物流更新，请帮忙核对发货与履约状态。",
            f"“{product}”这笔订单 {order_id} 已经等待多日，目前仍有{issue_hint} 的迹象，麻烦同步当前承运与履约进展。",
            f"关于订单 {order_id}，我们现在最关注的是“{product}”的发货与物流问题。由于存在{issue_hint}，请尽快确认是否需要升级处理。",
        ]
        return templates[idx % len(templates)]
    if category == "technical_issue":
        templates = [
            f"今天开始“{product}”持续出现{issue_hint}，多名同事反馈无法正常使用，已经影响团队工作推进，请尽快排查。",
            f"我们在使用“{product}”时遇到{issue_hint}，当前影响范围正在扩大，希望尽快确认故障根因和恢复时间。",
            f"“{product}”当前存在明显的{issue_hint}，现场已经很难继续推进业务，麻烦优先安排技术排查。",
        ]
        body = templates[idx % len(templates)]
        if customer_tier == "enterprise" and idx % 2 == 0:
            body += " 目前已经影响生产，SLA 有风险，请按高优先级处理。"
        return body
    if category == "account_access":
        templates = [
            f"我们在访问“{product}”时遇到{issue_hint}，已经尝试重置密码和重新验证，但仍然无法恢复，请协助排查账号或权限配置。",
            f"“{product}”当前存在{issue_hint}，用户多次尝试登录后仍然失败，希望尽快确认是账号、验证码还是 MFA 配置问题。",
            f"关于“{product}”的账号访问问题，目前表现为{issue_hint}，已经影响日常操作，请给出下一步处理建议。",
        ]
        return templates[idx % len(templates)]
    templates = [
        f"我们最近在评估“{product}”是否适合内部场景，想重点咨询与{issue_hint}相关的能力边界、上线节奏和最佳实践。",
        f"关于“{product}”，我们想进一步了解{issue_hint}方面的信息，尤其是适用范围、交付节奏和后续支持方式。",
        f"当前团队正在比较不同方案，希望确认“{product}”在{issue_hint}方面的能力和商务安排，请尽量给到清晰答复。",
    ]
    return templates[idx % len(templates)]


def _history_message(category: str, issue_hint: str, idx: int) -> str:
    templates = {
        "billing_refund": [
            f"已记录客户关于“{issue_hint}”的诉求，上一轮处理建议先核对订单与付款凭证。",
            f"历史备注显示该类退款工单通常需要先确认订单状态，再决定是否进入人工审批。",
        ],
        "delivery_issue": [
            f"已同步物流停滞情况，建议继续核对履约节点和承运状态。",
            f"历史记录显示此类工单应先确认“{issue_hint}”对应的订单状态，再考虑升级物流处理。",
        ],
        "technical_issue": [
            f"上一次处理该类故障时，先收集报错截图、影响范围和业务影响，再决定是否升级。",
            f"历史记录显示“{issue_hint}”相关问题往往需要优先排查系统性故障。",
        ],
        "account_access": [
            f"上一轮处理建议先核对密码重置、验证码和 MFA 配置。",
            f"历史备注显示账号访问异常通常需要先确认身份验证链路是否完整。",
        ],
        "general_inquiry": [
            f"历史备注显示此类咨询优先引用产品文档和商务资料，再给出功能边界说明。",
            f"上一轮处理给客户同步了产品能力概览，并标注了需要进一步确认的上线计划问题。",
        ],
    }
    return templates[category][idx % len(templates[category])]


def _history_message_from_public(
    category: str,
    issue_hint: str,
    idx: int,
    source_subject: str,
    source_body: str,
) -> str:
    source_excerpt = _public_excerpt(source_body, limit=140)
    templates = {
        "billing_refund": [
            f"公开同类工单“{source_subject}”提到：{source_excerpt}。上一轮处理建议优先核对订单与付款凭证，再决定是否进入退款审批。",
            f"历史案例参考“{source_subject}”，主要争议集中在{issue_hint}。处理时应先确认订单可核验，再评估退款或账单修正。",
        ],
        "delivery_issue": [
            f"公开同类工单“{source_subject}”描述了类似的履约问题：{source_excerpt}。历史经验是先同步订单状态，再判断是否需要升级物流处理。",
            f"历史案例“{source_subject}”显示{issue_hint}常与发货停滞或轨迹异常相关，建议先核对履约节点和承运状态。",
        ],
        "technical_issue": [
            f"公开同类工单“{source_subject}”出现了类似故障：{source_excerpt}。上一轮处置先收集影响范围、报错信息与业务影响，再决定是否升级。",
            f"历史技术案例“{source_subject}”显示{issue_hint}通常需要先排查系统性影响，再决定是否进入高优升级流程。",
        ],
        "account_access": [
            f"公开同类工单“{source_subject}”提到：{source_excerpt}。历史处理通常先核对密码重置、验证码和 MFA 配置链路。",
            f"历史访问异常案例“{source_subject}”表明{issue_hint}时应先确认身份验证链路，再判断是否属于系统性问题。",
        ],
        "general_inquiry": [
            f"公开同类咨询“{source_subject}”关注点为：{source_excerpt}。历史回复通常先引用产品资料，再说明能力边界与交付节奏。",
            f"参考公开咨询“{source_subject}”，当客户集中询问{issue_hint}时，建议先给功能概览，再标注需要进一步确认的上线安排。",
        ],
    }
    return templates[category][idx % len(templates[category])]


def build_seed_dataset(seed_dir: Path, *, random_seed: int = 7) -> None:
    random.seed(random_seed)
    now = datetime(2026, 3, 30, 9, 0, 0)
    public_corpus = _load_public_ticket_corpus(seed_dir)

    customers: list[dict[str, object]] = []
    for idx in range(60):
        tier = TIERS[idx % len(TIERS)]
        customers.append(
            {
                "customer_id": f"CUST-{idx + 1:03d}",
                "name": f"客户{idx + 1:03d}",
                "email": f"customer{idx + 1:03d}@example.com",
                "customer_tier": tier,
                "region": REGIONS[idx % len(REGIONS)],
                "loyalty_years": 1 + (idx % 8),
                "open_tickets": idx % 4,
            }
        )

    orders: list[dict[str, object]] = []
    for idx in range(120):
        delivered_days_ago = (idx % 45) + 1
        orders.append(
            {
                "order_id": f"ORD-{idx + 1:04d}",
                "customer_id": customers[idx % len(customers)]["customer_id"],
                "product": PRODUCTS[idx % len(PRODUCTS)],
                "status": "delivered" if idx % 5 else "pending",
                "delivered_days_ago": delivered_days_ago,
                "amount": round(199 + (idx % 13) * 39.5, 2),
                "eligible_for_refund": "true" if delivered_days_ago <= 30 and idx % 5 else "false",
            }
        )

    kb_rows: list[dict[str, object]] = []
    article_idx = 1
    for category, bodies in _kb_templates().items():
        for repetition in range(24):
            product = PRODUCTS[repetition % len(PRODUCTS)]
            kb_rows.append(
                {
                    "doc_id": f"KB-{article_idx:03d}",
                    "title": f"{CATEGORY_TITLES[category]}处理指引 {repetition + 1}",
                    "body": f"{bodies[repetition % len(bodies)]} 适用产品：{product}。",
                    "tags": f"{category},{product},support",
                    "product": product,
                    "category": category,
                }
            )
            article_idx += 1

    kb_rows = _public_kb_rows(public_corpus)

    policy_specs = [
        ("POL-001", "退款必须主管审批", "refund", True, "所有退款动作在执行前都必须进入人工审批。"),
        ("POL-002", "缺少订单号不能退款", "refund", False, "未核验订单号前，客服只能要求客户补充信息，不能承诺退款。"),
        ("POL-003", "企业 Sev1 故障必须升级", "escalation", True, "企业客户的 Sev1 生产故障必须由主管确认后升级给专家团队。"),
        ("POL-004", "禁止静默关单", "close_ticket_without_contact", True, "未与客户沟通前，工单不能直接静默关闭。"),
        ("POL-005", "SLA 特批需审批", "sla_override", True, "任何 SLA 例外都必须经过主管批准。"),
        ("POL-006", "物流长期延迟应升级", "escalation", False, "发货或运输状态停滞超过五天时，应创建物流升级单。"),
    ]
    policies: list[dict[str, object]] = []
    for idx, (policy_id, title, action_type, approval_required, body) in enumerate(policy_specs * 4):
        policies.append(
            {
                "policy_id": f"{policy_id}-{idx + 1:02d}",
                "title": title,
                "body": body,
                "action_type": action_type,
                "approval_required": "true" if approval_required else "false",
                "priority_hint": ["low", "medium", "high", "urgent"][idx % 4],
            }
        )

    reply_templates: list[dict[str, object]] = []
    reply_template_bodies = _reply_template_bodies()
    for idx in range(120):
        category = CATEGORY_SEQUENCE[idx % len(CATEGORY_SEQUENCE)]
        reply_templates.append(
            {
                "template_id": f"TPL-{idx + 1:03d}",
                "category": category,
                "tone": "concise" if idx % 2 else "warm",
                "body": reply_template_bodies[category],
            }
        )

    refundable_orders = [order for order in orders if order["eligible_for_refund"] == "true"]
    tickets: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    attachments: list[dict[str, object]] = []
    category_offsets = {category: 0 for category in CATEGORY_SEQUENCE}

    for idx in range(360):
        category = CATEGORY_SEQUENCE[idx % len(CATEGORY_SEQUENCE)]
        public_rows = public_corpus[category]
        source_row = public_rows[category_offsets[category] % len(public_rows)]
        category_offsets[category] += 1

        customer = customers[idx % len(customers)]
        created_at = now - timedelta(hours=idx % 96, minutes=idx % 53)
        issue_hint = _issue_hint(category, source_row["source_subject"], source_row["source_body"])

        if category == "billing_refund":
            order = refundable_orders[idx % len(refundable_orders)]
            missing_order = idx % 7 == 0
            linked_order_id = "" if missing_order else order["order_id"]
        elif category == "delivery_issue":
            order = orders[idx % len(orders)]
            linked_order_id = order["order_id"]
            missing_order = False
        else:
            order = orders[(idx * 3) % len(orders)]
            linked_order_id = order["order_id"] if idx % 4 == 0 else ""
            missing_order = False

        title = _ticket_title(category, issue_hint, idx)
        body = _ticket_body(
            category=category,
            issue_hint=issue_hint,
            product=order["product"],
            order_id=order["order_id"],
            missing_order=missing_order,
            customer_tier=customer["customer_tier"],
            idx=idx,
        )

        tickets.append(
            {
                "ticket_id": f"TCK-{idx + 1:04d}",
                "channel": CHANNELS[idx % len(CHANNELS)],
                "customer_id": customer["customer_id"],
                "customer_tier": customer["customer_tier"],
                "title": title,
                "body": body,
                "product": order["product"],
                "created_at": created_at.isoformat(),
                "status": "open",
                "linked_order_id": linked_order_id,
                "expected_category": category,
                "source_dataset": source_row["source_dataset"],
                "source_ticket_ref": source_row["corpus_id"],
                "source_language": source_row["source_language"],
                "source_queue": source_row["source_queue"],
                "source_subject": source_row["source_subject"],
                "source_body": source_row["source_body"],
            }
        )
        ticket_id = f"TCK-{idx + 1:04d}"
        if category == "billing_refund" and idx in {0, 5}:
            attachments.append(
                {
                    "attachment_id": f"ATT-SEED-{idx + 1:04d}",
                    "ticket_id": ticket_id,
                    "filename": "payment_success.png",
                    "file_type": "image",
                    "source_dataset": "synthetic_sroie_cord_style",
                    "storage_path": "",
                    "content_hash": f"seed-payment-{idx + 1:04d}",
                    "ocr_text": f"支付成功 订单号 {order['order_id']} 金额 {order['amount']} 元",
                    "visual_summary": "付款截图显示客户已经完成支付，适合作为退款诉求的辅助证据。",
                    "metadata": json.dumps({"public_source": "SROIE/CORD-style receipt simulation"}, ensure_ascii=False),
                    "parse_status": "pending",
                }
            )
        elif category == "technical_issue" and idx == 2:
            attachments.append(
                {
                    "attachment_id": f"ATT-SEED-{idx + 1:04d}",
                    "ticket_id": ticket_id,
                    "filename": "service_unavailable_screenshot.png",
                    "file_type": "image",
                    "source_dataset": "synthetic_rico_style",
                    "storage_path": "",
                    "content_hash": f"seed-error-{idx + 1:04d}",
                    "ocr_text": "Error 503 Service Unavailable 高级会员服务 页面无法加载",
                    "visual_summary": "报错截图显示服务不可用，页面提示 503。",
                    "metadata": json.dumps({"public_source": "RICO-style UI screenshot simulation"}, ensure_ascii=False),
                    "parse_status": "pending",
                }
            )
        elif category == "general_inquiry" and idx == 4:
            attachments.append(
                {
                    "attachment_id": f"ATT-SEED-{idx + 1:04d}",
                    "ticket_id": ticket_id,
                    "filename": "support_form.pdf",
                    "file_type": "pdf",
                    "source_dataset": "synthetic_docvqa_funsd_style",
                    "storage_path": "",
                    "content_hash": f"seed-form-{idx + 1:04d}",
                    "ocr_text": "服务申请表 产品 高级会员服务 日期 2026-03-30 处理时限 2 个工作日",
                    "visual_summary": "PDF 表单包含产品、日期和处理时限字段，可作为咨询类工单附件证据。",
                    "metadata": json.dumps({"public_source": "DocVQA/FUNSD-style document simulation"}, ensure_ascii=False),
                    "parse_status": "pending",
                }
            )
        histories.append(
            {
                "event_id": f"HIS-{idx + 1:04d}",
                "ticket_id": f"TCK-{idx + 1:04d}",
                "message": _history_message_from_public(
                    category,
                    issue_hint,
                    idx,
                    source_row["source_subject"],
                    source_row["source_body"],
                ),
                "created_at": created_at.isoformat(),
                "agent_name": f"agent_{idx % 9}",
            }
        )

    _write_rows(seed_dir / "customers.csv", list(customers[0].keys()), customers)
    _write_rows(seed_dir / "orders.csv", list(orders[0].keys()), orders)
    _write_rows(seed_dir / "kb_articles.csv", list(kb_rows[0].keys()), kb_rows)
    _write_rows(seed_dir / "policies.csv", list(policies[0].keys()), policies)
    _write_rows(seed_dir / "reply_templates.csv", list(reply_templates[0].keys()), reply_templates)
    _write_rows(seed_dir / "tickets.csv", list(tickets[0].keys()), tickets)
    _write_rows(seed_dir / "ticket_history.csv", list(histories[0].keys()), histories)
    if attachments:
        _write_rows(seed_dir / "ticket_attachments.csv", list(attachments[0].keys()), attachments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TicketFlow CSV seed data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "seed",
        help="Directory where the CSV seed files will be written.",
    )
    args = parser.parse_args()
    build_seed_dataset(args.output_dir)
    print(f"Seed data written to {args.output_dir}")


if __name__ == "__main__":
    main()
