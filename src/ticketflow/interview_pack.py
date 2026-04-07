from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import run_demo
from .evals import run_evaluation


def _pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def _latency(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f} ms"


def _load_minimind_report(project_root: Path) -> dict | None:
    report_path = project_root.parent / "minimind-ticket" / "reports" / "ticket_base_vs_tuned_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_interview_pack(
    project_root: Path,
    sample_size: int = 30,
    model_backend: str = "rule",
) -> str:
    overrides = {"MODEL_BACKEND": model_backend}
    report = run_evaluation(project_root, sample_size=sample_size, overrides=overrides)
    scenarios = run_demo(project_root, reset_data=True, overrides=overrides)
    minimind_report = _load_minimind_report(project_root)

    scenario_lines = []
    for scenario in scenarios:
        scenario_lines.append(
            "\n".join(
                [
                    f"### {scenario['scenario']}",
                    f"- 工单标题：{scenario['title']}",
                    f"- 客户等级：{scenario['customer_tier']}",
                    f"- 预测类别：{scenario['predicted_category']}",
                    f"- 分诊来源：{scenario['triage_source']}",
                    f"- 建议动作：{scenario['proposed_action']}",
                    f"- 动作来源：{scenario['action_source']}",
                    f"- 审批状态：{scenario['approval_state']}",
                    f"- 最终状态：{scenario['final_status']}",
                    f"- 回复来源：{scenario['reply_source']}",
                    f"- 回复预览：{scenario['customer_reply_preview']}",
                ]
            )
        )

    bad_case_lines = []
    for case in report.get("bad_case_examples", []):
        bad_case_lines.append(
            f"- {case['ticket_id']} | {case['kind']} | expected={case['expected']} | predicted={case['predicted']} | {case['title']}"
        )
    if not bad_case_lines:
        bad_case_lines.append("- 当前样本中未发现分类错误案例。")

    minimind_lines = []
    if minimind_report is None:
        minimind_lines.append("- 当前还没有生成 `MiniMind-Ticket` 的 base vs tuned 对比报告，可在 LoRA 训练完成后补齐。")
    else:
        minimind_lines.extend(
            [
                f"- 基座模型总体得分：{minimind_report['base']['overall_score']}",
                f"- 微调模型总体得分：{minimind_report['tuned']['overall_score']}",
                f"- 总体相对提升：{_pct(minimind_report['overall_gain_pct'])}",
            ]
        )
        for task_name, task_payload in minimind_report.get("task_gain", {}).items():
            minimind_lines.append(
                f"- {task_name}：base={task_payload['base']} / tuned={task_payload['tuned']} / 提升={_pct(task_payload['gain_pct'])}"
            )

    return f"""# TicketFlow 中文面试包

## 项目一句话
TicketFlow 是一个面向客服 / IT 服务台场景的企业工单协同 Agent。它把 LangGraph 工作流、RAG 检索、MCP 外部协同和可切换模型后端串成一个可追踪、可量化、可演示的完整闭环。

## 四层架构
- `Agent workflow`：负责分诊、审批、执行、审计和回退。
- `RAG`：在 `retrieve_context` 节点检索知识库、策略和历史工单，为后续决策提供证据。
- `MCP`：通过独立邮件 MCP Server 调用真实 SMTP 邮箱，完成升级通知和知识候选沉淀。
- `LoRA`：由 `MiniMind-Ticket` 负责中文工单领域微调，并作为可切换的本地模型后端。

## 模块量化指标
### 1. Agent workflow
- 样本数：{report['sample_size']}
- 分类准确率：{_pct(report['classification_accuracy'])}
- 升级判断 F1：{report['escalation_f1']:.3f}
- 敏感动作审批拦截率：{_pct(report['sensitive_action_approval_interception_rate'])}
- 首轮回复可用率：{_pct(report['first_reply_usable_rate'])}
- 不受支持的敏感动作误执行率：{_pct(report['unsupported_sensitive_action_rate'])}
- 平均处理步数：{report['average_processing_steps']}

### 2. RAG
- 策略命中率：{_pct(report['rag_policy_hit_rate'])}，相对基线提升：{_pct(report['policy_hit_lift_pct'])}
- 知识库命中率：{_pct(report['rag_kb_hit_rate'])}，相对基线提升：{_pct(report['kb_hit_lift_pct'])}
- 历史案例命中率：{_pct(report['rag_history_hit_rate'])}，相对基线提升：{_pct(report['history_hit_lift_pct'])}
- 向量检索参与率：{_pct(report['vector_retrieval_usage_rate'])}
- 证据引用覆盖率：{_pct(report['citation_grounding_rate'])}

### 3. MCP 外部协同
- 升级通知触发率：{_pct(report['incident_email_trigger_rate'])}
- 升级通知邮件成功率：{_pct(report['incident_email_success_rate'])}
- 知识候选触发率：{_pct(report['kb_candidate_trigger_rate'])}
- 知识候选邮件成功率：{_pct(report['kb_candidate_success_rate'])}
- 外部调用非阻塞恢复率：{_pct(report['external_nonblocking_recovery_rate'])}
- 外部邮件平均耗时：{_latency(report['external_delivery_avg_latency_ms'])}

### 4. 模型后端
- 云模型分诊参与率：{_pct(report['triage_cloud_rate'])}
- MiniMind 分诊参与率：{_pct(report['triage_minimind_rate'])}
- 云模型回复参与率：{_pct(report['reply_cloud_rate'])}
- MiniMind 回复参与率：{_pct(report['reply_minimind_rate'])}
- 整体回退案例占比：{_pct(report['fallback_case_rate'])}

## LoRA / MiniMind-Ticket
{chr(10).join(minimind_lines)}

## 面试讲法
- 这不是普通聊天机器人，而是一个面向客服 / IT 服务台场景的业务 Agent。
- LangGraph 负责状态化工作流；RAG 负责把知识证据拉进上下文；MCP 负责对接真实外部工具；LoRA 负责把模型做成更懂工单语境的后端。
- 高风险动作不会直接执行，而是经过审批中断；外部通知失败也不会阻塞主流程，只会留下审计和 trace。

## 代表性场景
{chr(10).join(scenario_lines)}

## 风险与坏例子
{chr(10).join(bad_case_lines)}

## 简历写法
- 基于 LangGraph 构建企业工单协同 Agent，完成分诊、混合 RAG 检索、策略约束动作建议、人工审批与工具执行闭环。
- 将升级通知与知识候选沉淀能力标准化为 MCP tools，通过真实 SMTP 邮件系统完成外部协同，并保证失败时主流程非阻塞。
- 设计可量化评测体系，分别评估 Agent workflow、RAG 命中提升、MCP 外部协同成功率以及云模型 / MiniMind 后端参与率与回退率。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Chinese interview pack for TicketFlow.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TicketFlow 项目根目录。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="离线评测样本数。",
    )
    parser.add_argument(
        "--model-backend",
        choices=["rule", "cloud_api", "minimind_api", "minimind_split"],
        default="rule",
        help="生成面试包时使用的模型后端。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "interview_pack_cn.md",
        help="输出 markdown 文件路径。",
    )
    args = parser.parse_args()

    content = build_interview_pack(
        args.project_root,
        sample_size=args.sample_size,
        model_backend=args.model_backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Interview pack written to {args.output}")


if __name__ == "__main__":
    main()
