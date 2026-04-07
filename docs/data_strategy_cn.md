# TicketFlow 数据策略说明

## 核心原则
项目的数据来源分成两层：

- `公开真实文本层`：优先使用公开 support / helpdesk 数据集中的真实工单文本，提升项目的可信度。
- `业务侧结构化表`：继续保留本地构造的订单、客户、策略、审批与外部协同相关表，用于支撑完整 workflow。

这套策略的目标不是追求“全真实生产数据”，而是做到：

- 工单主文本尽量来自真实公开语料
- 业务闭环保持完整
- 面试时能清楚解释哪些数据来自公开数据集，哪些是为了业务流程而补充构造

## 已优先公开化的部分

### 1. 工单标题与正文
- `tickets.csv`

当前工单主文本优先来自缓存的公开 support-ticket 语料：
- `Tobi-Bueck/customer-support-tickets`

项目会保留以下公开来源字段：
- `source_dataset`
- `source_ticket_ref`
- `source_language`
- `source_queue`
- `source_subject`
- `source_body`

同时生成中文业务映射文本，保证页面演示和 Agent workflow 仍贴合国内工单场景。

### 2. 知识库摘要
- `kb_articles.csv`

当前知识库不再完全由本地模板生成，而是优先根据公开工单原文生成案例摘要型 KB 条目，再交给本地 RAG 检索层使用。

### 3. 历史工单备注
- `ticket_history.csv`

历史记录文本优先基于公开工单原文摘录生成，避免完全脱离真实 ticket 语料。

### 4. MiniMind 训练 / 评测输入文本
- `minimind-ticket/scripts/build_ticket_dataset.py`

MiniMind 的分类、决策、回复、内部备注任务在构造 prompt 时，优先注入：
- 公开工单原始 subject/body
- 对应的中文业务映射 title/body

这样可以让本地模型的训练与评测更多基于真实公开 ticket 文本，而不是只看本地改写后的模板化语料。

## 继续保留本地构造的部分

### 1. 订单表
- `orders.csv`

原因：
- 公开 helpdesk 数据通常不包含订单主数据
- 退款、履约、物流判断必须依赖订单字段

### 2. 客户画像表
- `customers.csv`

原因：
- `customer_tier`
- 区域
- 历史工单活跃度
- 企业客户标记

这些字段通常不在公开工单语料中，但对优先级和 SLA 风险判断很重要。

### 3. 策略规则表
- `policies.csv`

原因：
- 审批逻辑
- 退款约束
- SLA 特批
- 禁止静默关单

这类内容本来就是企业内部规则，不适合伪装成“公开真实数据”。

### 4. 外部协同表
- `external_email_deliveries`
- `knowledge candidate` 相关记录

原因：
- 这些属于系统运行结果和运营状态
- 公开数据集通常不会包含

## 当前项目的合理定位
现在的项目不是：

- `全生产级真实数据系统`

而是：

- `公开真实工单文本 + 本地业务侧表` 的工单 Agent 原型系统

这个定位是合理的，因为它同时保留了：

- 真实语料带来的可信度
- 完整 workflow 所需的结构化业务上下文
- RAG / MCP / LoRA 的可验证链路

## 面试推荐说法

> 工单和客服文本主语料来自公开 support / helpdesk 数据集，我在此基础上做了中文业务映射；为了验证退款、审批、外部协同和知识沉淀这些业务闭环，我补充构造了订单、客户、策略和历史记录等结构化业务侧表。也就是说，文本主输入尽量来自真实公开语料，业务流程约束则由本地结构化表支撑。

## 后续迭代优先级
1. 继续扩大真实公开工单文本覆盖率
2. 为 MiniMind 补充真实中文客服 / 帮助台回复评测集
3. 增加 harder-case 与 OOD 样本
4. 降低模板式 reply / internal_note 在训练集中的占比
