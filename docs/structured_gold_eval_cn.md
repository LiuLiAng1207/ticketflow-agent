# 结构化 Gold 评测说明

## 为什么要单独做这一层
当前 `TicketFlow / MiniMind-Ticket` 里的结构化任务如果只用启发式标签训练、再用同生成器切分测试集来评，很容易高估模型表现。

为了让项目更接近开源 benchmark 的常见做法，结构化评测应拆成两层：

- `训练层`：允许使用启发式/规则生成的数据来放大样本量，帮助模型学会业务 schema。
- `评测层`：使用真实公开 ticket 文本 + 小规模人工 gold 标注，避免直接“用自己定义的规则评自己”。

## 当前脚手架做了什么
项目新增了两条命令：

```powershell
ticketflow-build-gold-structured-eval
ticketflow-eval-structured-gold
```

### 1. 生成待标注集
命令：

```powershell
cd D:\Study\shixi\ticketflow
ticketflow-build-gold-structured-eval --sample-size 40
```

默认输出：

`D:\Study\shixi\ticketflow\data\gold_eval\public_structured_gold_template.csv`

这个文件的来源不是纯本地编造，而是：

- 工单文本：来自公开 support ticket 语料映射后的 `tickets.csv`
- 业务侧字段：沿用 workflow 所需的 `customer_tier / linked_order_id / product`
- `silver_*` 字段：当前系统给出的候选标签，只作为标注参考，不是 gold

### 2. 人工标注
需要人工填写以下字段：

- `gold_category`
- `gold_priority`
- `gold_sla_risk`
- `gold_action_type`
- `gold_approval_required`
- `gold_target_status`

同时把：

- `annotation_status`

改成：

- `annotated`
- `reviewed`
- 或 `final`

## 标注口径建议

### `gold_category`
只允许以下值：

- `billing_refund`
- `delivery_issue`
- `technical_issue`
- `account_access`
- `general_inquiry`

### `gold_priority`
只允许：

- `low`
- `medium`
- `high`
- `urgent`

### `gold_sla_risk`
只允许：

- `true`
- `false`

推荐标准：

- 如果文本里出现明显的生产阻塞、业务中断、严重故障、高优响应承诺，则标 `true`
- 普通咨询、常规物流、一般账单问题通常标 `false`

### `gold_action_type`
只允许：

- `refund`
- `escalation`
- `request_info`
- `status_update`
- `troubleshoot`
- `sla_override`
- `close_ticket_without_contact`

### `gold_approval_required`
只允许：

- `true`
- `false`

高风险动作建议默认需要审批：

- `refund`
- `escalation`
- `sla_override`
- `close_ticket_without_contact`

### `gold_target_status`
推荐使用以下状态集合：

- `open`
- `in_progress`
- `investigating`
- `monitoring`
- `waiting_on_customer`
- `pending_finance`
- `pending_human`
- `escalated`

## 跑 gold 评测
标注完成后运行：

```powershell
cd D:\Study\shixi\ticketflow
ticketflow-eval-structured-gold `
  --gold-path D:\Study\shixi\ticketflow\data\gold_eval\public_structured_gold_template.csv `
  --model-backend rule `
  --report-path D:\Study\shixi\ticketflow\docs\structured_gold_eval_report.json
```

当前脚本会输出这些核心指标：

- `category_accuracy`
- `category_macro_f1`
- `priority_accuracy`
- `sla_risk_recall`
- `action_type_accuracy`
- `approval_required_recall`
- `approval_required_precision`
- `approval_required_f1`
- `target_status_accuracy`

## 面试时怎么讲
推荐说法：

> 训练阶段我保留了启发式标签来扩大样本量，但最终结构化任务评测采用真实公开 ticket 文本上的小规模人工 gold 标注集合。这样可以把“模型学会 schema”与“模型在真实 ticket 上是否真的判断正确”分开看，避免直接用同源规则评测自己。
