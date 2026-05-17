# TicketFlow 多模态工单 V1 设计说明

## 定位

TicketFlow 多模态 V1 不做“看图聊天机器人”，而是把截图、PDF、票据、表单等附件先解析成可审计证据，再进入现有的 RAG、证据充分性判断、动作路由、工具审批和回复事实校验链路。

核心链路：

```text
文本工单 + 附件
-> 附件解析
-> 附件证据化
-> 多路上下文检索
-> 证据充分性判断
-> 动作路由/审批
-> 回复事实校验
```

## 公开数据集口径

目前没有一个公开数据集能直接覆盖“客服/IT 工单 + 截图/PDF/收据/日志附件 + 动作标签”。V1 采用组合构造口径：

- 客服/IT 工单文本：`Tobi-Bueck/customer-support-tickets`、Kaggle Customer Support Ticket Dataset。
- 支付截图/收据：SROIE、CORD，用于模拟支付截图、收据和退款附件。
- PDF/表单附件：DocVQA、FUNSD，用于模拟表单、PDF 和扫描件证据。
- UI/报错截图：RICO 或 RICO 衍生数据，用于模拟页面报错、登录失败、服务不可用截图。
- 日志附件：Loghub 作为 V1.5 扩展，用于系统日志和异常日志工单。

这些数据源只作为 benchmark/demo 样本来源，动作标签、证据充分性标签和业务治理标签需要二次构造，不能直接说公开数据集原生提供了完整工单处理标签。

## 当前实现

V1 新增两类核心结构：

- `AttachmentRecord`：保存附件元信息、OCR 文本、视觉摘要、来源、hash 和解析状态。
- `AttachmentEvidence`：保存附件证据类型、抽取文本、视觉摘要、结构化字段、置信度和风险标记。

LangGraph 新链路：

```text
intake
-> parse_attachments
-> triage
-> retrieve_context
-> assess_sufficiency
-> route_action
-> propose_action
-> approval_gate
-> execute_action
-> draft_reply
-> fact_check_reply
-> finalize
```

`parse_attachments` 节点在分诊前运行，这样支付截图、报错截图和 PDF 表单中的信息可以作为后续分诊、检索和充分性判断的上下文补充。

## 附件解析策略

当前 V1 使用确定性解析器，优先处理已经得到的 OCR 文本和视觉摘要：

- 支付/收据截图：抽取 `order_id`、`amount`、`payment_status` 等字段。
- 报错截图：抽取 `error_code`、页面提示和服务状态。
- PDF/表单：抽取表单字段、日期、产品和处理时限。
- 低置信度附件：只作为弱证据展示，不单独放行高风险动作。

后续可以替换为 OCR 服务、Document AI、视觉大模型或多模态 embedding，但替换后的输出仍应落到 `AttachmentEvidence`，避免破坏治理链。

## 治理边界

附件只能增强证据，不能替代核心业务事实。

- 退款动作仍优先要求 `policy + order`。
- 支付截图只能辅助说明客户诉求，不能单独证明订单系统真实有效。
- 如果订单系统缺失但截图显示已支付，系统应走 `request_info` 或 `pending_human`，不能直接退款。
- 回复中出现“截图显示”“附件证明”“PDF 中写到”等 claim 时，必须能追溯到 `attachment_id/evidence_id`。

## 评测指标

V1 建议新增：

- 附件解析成功率。
- OCR 字段抽取 F1。
- 金额、订单号、错误码抽取准确率。
- 多模态证据召回率（multimodal evidence recall@K）。
- 附件证据 MRR。
- 多模态证据充分性准确率。
- 附件不足时保守降级率。
- 附件 claim 有据率。
- unsupported attachment claim rate。

这些指标需要真实专项评测跑完后再填数字，不能提前写新分数。

## 面试讲法

可以这样说：

> 我把多模态工单处理设计成“附件先证据化，再进入治理链”。截图、PDF、票据不会直接交给大模型自由决策，而是先经过 OCR、视觉摘要和字段抽取，生成 `AttachmentEvidence`。这些证据和原来的 policy、order、history 一起进入 RAG 和证据充分性判断。附件可以增强上下文，但高风险动作仍然必须经过策略、订单和审批链路，避免因为一张截图就错误退款或越权处理。
