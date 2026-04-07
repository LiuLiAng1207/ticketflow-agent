# TicketFlow 中文面试包

## 项目一句话
TicketFlow 是一个面向客服 / IT 服务台场景的企业工单协同 Agent。它把 LangGraph 工作流、RAG 检索、MCP 外部协同和可切换模型后端串成一个可追踪、可量化、可演示的完整闭环。

## 四层架构
- `Agent workflow`：负责分诊、审批、执行、审计和回退。
- `RAG`：在 `retrieve_context` 节点检索知识库、策略和历史工单，为后续决策提供证据。
- `MCP`：通过独立邮件 MCP Server 调用真实 SMTP 邮箱，完成升级通知和知识候选沉淀。
- `LoRA`：由 `MiniMind-Ticket` 负责中文工单领域微调，并作为可切换的本地模型后端。

## 模块量化指标
### 1. Agent workflow
- 样本数：12
- 分类准确率：100.0%
- 升级判断 F1：0.667
- 敏感动作审批拦截率：100.0%
- 首轮回复可用率：100.0%
- 不受支持的敏感动作误执行率：0.0%
- 平均处理步数：9.0

### 2. RAG
- 策略命中率：100.0%，相对基线提升：100.0%
- 知识库命中率：100.0%，相对基线提升：0.0%
- 历史案例命中率：100.0%，相对基线提升：100.0%
- 向量检索参与率：0.0%
- 证据引用覆盖率：100.0%

### 3. MCP 外部协同
- 升级通知触发率：33.3%
- 升级通知邮件成功率：0.0%
- 知识候选触发率：66.7%
- 知识候选邮件成功率：0.0%
- 外部调用非阻塞恢复率：100.0%
- 外部邮件平均耗时：0.0 ms

### 4. 模型后端
- 云模型分诊参与率：0.0%
- MiniMind 分诊参与率：0.0%
- 云模型回复参与率：0.0%
- MiniMind 回复参与率：0.0%
- 整体回退案例占比：0.0%

## LoRA / MiniMind-Ticket
- 当前还没有生成 `MiniMind-Ticket` 的 base vs tuned 对比报告，可在 LoRA 训练完成后补齐。

## 面试讲法
- 这不是普通聊天机器人，而是一个面向客服 / IT 服务台场景的业务 Agent。
- LangGraph 负责状态化工作流；RAG 负责把知识证据拉进上下文；MCP 负责对接真实外部工具；LoRA 负责把模型做成更懂工单语境的后端。
- 高风险动作不会直接执行，而是经过审批中断；外部通知失败也不会阻塞主流程，只会留下审计和 trace。

## 代表性场景
### refund_approval
- 工单标题：想申请最近订单的退款
- 客户等级：enterprise
- 预测类别：billing_refund
- 分诊来源：rule
- 建议动作：refund
- 动作来源：rule
- 审批状态：approved
- 最终状态：pending_finance
- 回复来源：rule
- 回复预览：我们已经根据订单 ORD-0004 发起退款审核，财务确认后会继续同步进展。
### missing_order_guardrail
- 工单标题：想申请最近订单的退款
- 客户等级：standard
- 预测类别：billing_refund
- 分诊来源：rule
- 建议动作：request_info
- 动作来源：rule
- 审批状态：not_required
- 最终状态：waiting_on_customer
- 回复来源：rule
- 回复预览：为了继续安全处理，请补充订单号、付款截图或其他可核验信息。
### enterprise_escalation
- 工单标题：系统故障影响团队使用
- 客户等级：enterprise
- 预测类别：technical_issue
- 分诊来源：rule
- 建议动作：escalation
- 动作来源：rule
- 审批状态：approved
- 最终状态：escalated
- 回复来源：rule
- 回复预览：当前问题已升级给专家团队处理，我们会基于现有上下文继续推进，并尽快反馈。
### delivery_monitoring
- 工单标题：订单一直没有发货
- 客户等级：standard
- 预测类别：delivery_issue
- 分诊来源：rule
- 建议动作：escalation
- 动作来源：rule
- 审批状态：approved
- 最终状态：escalated
- 回复来源：rule
- 回复预览：当前问题已升级给专家团队处理，我们会基于现有上下文继续推进，并尽快反馈。

## 风险与坏例子
- 当前样本中未发现分类错误案例。

## 简历写法
- 基于 LangGraph 构建企业工单协同 Agent，完成分诊、混合 RAG 检索、策略约束动作建议、人工审批与工具执行闭环。
- 将升级通知与知识候选沉淀能力标准化为 MCP tools，通过真实 SMTP 邮件系统完成外部协同，并保证失败时主流程非阻塞。
- 设计可量化评测体系，分别评估 Agent workflow、RAG 命中提升、MCP 外部协同成功率以及云模型 / MiniMind 后端参与率与回退率。
