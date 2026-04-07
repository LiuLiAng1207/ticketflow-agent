from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .evals import run_evaluation


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _compare_metric(before: float, after: float) -> dict[str, float | None]:
    absolute_delta = after - before
    relative_gain_pct = None if before == 0 else ((after - before) / before) * 100
    return {
        "before": _round(before),
        "after": _round(after),
        "absolute_delta": _round(absolute_delta),
        "relative_gain_pct": None if relative_gain_pct is None else round(relative_gain_pct, 1),
    }


def _load_minimind_report(project_root: Path) -> dict[str, Any] | None:
    candidates = [
        project_root.parent / "minimind-ticket" / "reports" / "ticket_base_vs_tuned_report_v1.json",
        project_root.parent / "minimind-ticket" / "reports" / "ticket_base_vs_tuned_report.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _pct(value: float | None) -> str:
    if value is None:
        return "不适用"
    return f"{value * 100:.1f}%"


def _delta_text(metric: dict[str, Any]) -> str:
    before = metric.get("before")
    after = metric.get("after")
    delta = metric.get("absolute_delta")
    rel = metric.get("relative_gain_pct")
    if before is None or after is None:
        return "无可用对比数据"
    if rel is None:
        return f"{_pct(before)} -> {_pct(after)}，绝对提升 {delta:.3f}"
    return f"{_pct(before)} -> {_pct(after)}，绝对变化 {delta:.3f}，相对变化 {rel:.1f}%"


def _localize_module_report(report: dict[str, Any]) -> dict[str, Any]:
    agent = report["agent_workflow"]
    rag = report["rag"]
    mcp = report["mcp"]
    lora_offline = report["lora_offline"]
    lora_in_flow = report["lora_in_ticketflow"]
    minimind_reply_after = lora_in_flow["reply_minimind_coverage_rate"].get("after")

    return {
        "项目": "TicketFlow / MiniMind-Ticket 四层量化报告",
        "评测口径": {
            "方法原则": report["methodology"]["principles"],
            "RAG 对比方式": report["methodology"]["rag"],
            "MCP 对比方式": report["methodology"]["mcp"],
            "LoRA 对比方式": report["methodology"]["lora"],
        },
        "总体说明": [
            "当前高分主要反映在“公开 support ticket 语料中文化改写 + 自建业务侧表”评测集上的流程正确性与安全性，不等价于真实生产环境准确率。",
            "主工作流中的分类节点默认以规则链为主，因此分类准确率偏高是符合当前实现的，不是云模型或本地模型在开放场景上的真实泛化能力。",
            "RAG 的 100% 命中率表示在当前评测样本上至少命中了一条相关策略或历史案例，不表示真实企业知识库下的召回率已经饱和。",
            "MCP 邮件成功率当前基于本地 SMTP sink 验证真实协议链路，说明工具调用和非阻塞回退逻辑可信，但不代表公网邮箱送达率。",
            "MiniMind 微调结果来自同分布中文 held-out 集，说明领域适配有效，但仍需更难样本和人工评审来验证泛化能力。",
            "LoRA 部分的结构化指标已拆分为“严格 JSON 有效率”和“可恢复 JSON 有效率”，用于区分模型真正学会稳定按 schema 输出，还是仅仅输出了可修复的代码块/半结构化结果。",
        ],
        "Agent 工作流": {
            "分类准确率": _pct(agent["classification_accuracy"]),
            "升级判断 F1": round(agent["escalation_f1"], 3),
            "敏感动作审批拦截率": _pct(agent["approval_interception_rate"]),
            "首轮回复可用率": _pct(agent["first_reply_usable_rate"]),
            "不支持敏感动作误执行率": _pct(agent["unsupported_sensitive_action_rate"]),
            "解释": "分类准确率高，主要因为当前评测集采用公开 support ticket 语料改写后的中文工单，并与本地业务侧表绑定；同时 triage 默认走规则链，所以这组结果更能说明流程稳定性，而不是开放域模型能力。",
        },
        "RAG 模块": {
            "策略命中率": _delta_text(rag["policy_hit_rate"]),
            "历史案例命中率": _delta_text(rag["history_hit_rate"]),
            "知识条目命中率": _delta_text(rag["kb_hit_rate"]),
            "向量检索参与率": _pct(rag["vector_retrieval_usage_rate"]),
            "证据引用覆盖率": _pct(rag["citation_grounding_rate"]),
            "解释": "这里的提升口径是“混合检索”相对“只查知识库”的提升，所以策略和历史案例从 0 到 100% 是合理的，因为基线本来就不检索这些源。",
        },
        "MCP 外部协同": {
            "升级通知成功率": _delta_text(mcp["incident_delivery_success_rate"]),
            "知识候选提交成功率": _delta_text(mcp["kb_candidate_success_rate"]),
            "外部失败非阻塞恢复率": _delta_text(mcp["external_nonblocking_recovery_rate"]),
            "升级通知触发率": _pct(mcp["enabled_trigger_rates"]["incident_email_trigger_rate"]),
            "知识候选触发率": _pct(mcp["enabled_trigger_rates"]["kb_candidate_trigger_rate"]),
            "平均外部调用耗时": f"{mcp['enabled_avg_latency_ms']:.1f} ms",
            "解释": "这里的成功率来自开启本地 SMTP sink 后的真实协议发送链路，反映的是接口集成质量和主流程的容错能力。",
        },
        "LoRA 离线微调": {
            "样本数": lora_offline.get("sample_size"),
            "总体得分": _delta_text(lora_offline["overall_score"]) if "overall_score" in lora_offline else "暂无",
            "严格 JSON 有效率": _delta_text(lora_offline["strict_json_valid_rate"]) if "strict_json_valid_rate" in lora_offline else "暂无",
            "可恢复 JSON 有效率": _delta_text(lora_offline["recoverable_json_rate"]) if "recoverable_json_rate" in lora_offline else "暂无",
            "分类任务得分": _delta_text(lora_offline["tasks"]["classification"]) if lora_offline.get("tasks") else "暂无",
            "决策任务得分": _delta_text(lora_offline["tasks"]["decision"]) if lora_offline.get("tasks") else "暂无",
            "内部备注得分": _delta_text(lora_offline["tasks"]["internal_note"]) if lora_offline.get("tasks") else "暂无",
            "客户回复得分": _delta_text(lora_offline["tasks"]["reply"]) if lora_offline.get("tasks") else "暂无",
            "解析方式分布": lora_offline.get("parse_mode_breakdown", {}),
            "失败原因分布": lora_offline.get("bad_case_reason_breakdown", {}),
            "解释": "MiniMind base 模型在这个领域上初始表现很弱，所以相对提升会很大。更合理的讲法是：微调前很多样本已经能被代码块修复或截取 JSON 子串后解析，但原样输出不稳定；微调后严格 JSON 和可恢复 JSON 同时提升，说明结构化输出真正被训稳了。",
        },
        "LoRA 接入主项目": {
            "本地模型回复覆盖率": _delta_text(lora_in_flow["reply_minimind_coverage_rate"]),
            "首轮回复可用率变化": _delta_text(lora_in_flow["first_reply_usable_rate"]),
            "回退案例占比变化": _delta_text(lora_in_flow["fallback_case_rate"]),
            "分类准确率变化": _delta_text(lora_in_flow["classification_accuracy"]),
            "解释": (
                "当前只把 MiniMind 设计成回复节点的可切换后端，没有替换分类节点，所以主流程稳定性保持不变。"
                if minimind_reply_after and minimind_reply_after > 0
                else "当前离线总报告生成时，本地 MiniMind 回复端点没有成功接住请求，因此覆盖率仍为 0。这个结果应该被如实保留，不能夸大成本地模型已经稳定接管主流程。"
            ),
        },
        "面试建议": [
            "不要把 100% 分类准确率说成真实线上准确率，要明确说这是“公开 support ticket 语料中文化改写 + 本地业务侧表”上的流程评测结果。",
            "RAG 部分重点讲‘为什么能从 0 提升到 100%’，因为基线只查知识库，不查策略和历史案例；这体现的是检索源扩展带来的增益。",
            "MCP 部分重点讲非阻塞恢复和外部链路标准化，不要夸大成真实业务邮箱运营指标。",
            "LoRA 部分重点讲严格 JSON、有代码块修复后的可恢复 JSON，以及领域适配效果，不要直接说模型已经达到生产级客服质量。",
        ],
    }


def _module_report_markdown(localized: dict[str, Any]) -> str:
    lines = [
        "# TicketFlow / MiniMind-Ticket 量化结果说明",
        "",
        "## 评测口径",
    ]
    for item in localized["评测口径"]["方法原则"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            f"- RAG：{localized['评测口径']['RAG 对比方式']}",
            f"- MCP：{localized['评测口径']['MCP 对比方式']}",
            f"- LoRA：{localized['评测口径']['LoRA 对比方式']}",
            "",
            "## 总体说明",
        ]
    )
    for item in localized["总体说明"]:
        lines.append(f"- {item}")
    for section in ("Agent 工作流", "RAG 模块", "MCP 外部协同", "LoRA 离线微调", "LoRA 接入主项目"):
        lines.extend(["", f"## {section}"])
        for key, value in localized[section].items():
            if key == "解释":
                lines.append(f"- 解释：{value}")
            else:
                lines.append(f"- {key}：{value}")
    lines.extend(["", "## 面试建议"])
    for item in localized["面试建议"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _start_smtp_sink(project_root: Path, host: str, port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ticketflow.smtp_sink",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=project_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _build_lora_offline_section(minimind_report: dict[str, Any] | None) -> dict[str, Any]:
    if minimind_report is None:
        return {
            "status": "missing_report",
            "message": "未找到 MiniMind base vs tuned 评测报告，请先运行 minimind-ticket 的评测脚本。",
        }

    base = minimind_report["base"]
    tuned = minimind_report["tuned"]
    tasks = sorted(set(base["by_task"]) | set(tuned["by_task"]))
    task_metrics = {
        task: _compare_metric(base["by_task"].get(task, 0.0), tuned["by_task"].get(task, 0.0))
        for task in tasks
    }
    return {
        "sample_size": base["sample_size"],
        "overall_score": _compare_metric(base["overall_score"], tuned["overall_score"]),
        "strict_json_valid_rate": _compare_metric(
            base.get("strict_json_valid_rate", base.get("json_valid_rate", 0.0)),
            tuned.get("strict_json_valid_rate", tuned.get("json_valid_rate", 0.0)),
        ),
        "recoverable_json_rate": _compare_metric(
            base.get("recoverable_json_rate", base.get("json_valid_rate", 0.0)),
            tuned.get("recoverable_json_rate", tuned.get("json_valid_rate", 0.0)),
        ),
        "parse_mode_breakdown": {
            "base": base.get("parse_mode_breakdown", {}),
            "tuned": tuned.get("parse_mode_breakdown", {}),
        },
        "bad_case_reason_breakdown": {
            "base": base.get("bad_case_reason_breakdown", {}),
            "tuned": tuned.get("bad_case_reason_breakdown", {}),
        },
        "tasks": task_metrics,
    }


def build_report(project_root: Path, sample_size: int = 12) -> dict[str, Any]:
    rule_report = run_evaluation(
        project_root,
        sample_size=sample_size,
        overrides={"MODEL_BACKEND": "rule"},
    )

    mcp_disabled = run_evaluation(
        project_root,
        sample_size=sample_size,
        overrides={
            "MODEL_BACKEND": "rule",
            "INCIDENT_EMAIL_TO": "",
            "KB_OPS_EMAIL_TO": "",
            "SMTP_HOST": "",
            "SMTP_USERNAME": "",
            "SMTP_AUTH_CODE": "",
        },
    )

    smtp_host = "127.0.0.1"
    smtp_port = 1025
    sink = _start_smtp_sink(project_root, smtp_host, smtp_port)
    time.sleep(2)
    try:
        mcp_enabled = run_evaluation(
            project_root,
            sample_size=sample_size,
            overrides={
                "MODEL_BACKEND": "rule",
                "SMTP_HOST": smtp_host,
                "SMTP_PORT": str(smtp_port),
                "SMTP_USERNAME": "ticketflow-local@example.com",
                "SMTP_AUTH_CODE": "",
                "SMTP_USE_TLS": "false",
                "EMAIL_FROM_NAME": "TicketFlow Agent",
                "INCIDENT_EMAIL_TO": "incident@test.local",
                "KB_OPS_EMAIL_TO": "knowledge@test.local",
            },
        )
    finally:
        sink.terminate()
        try:
            sink.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sink.kill()

    minimind_integration = run_evaluation(
        project_root,
        sample_size=sample_size,
        overrides={
            "MODEL_BACKEND": "minimind_api",
            "MINIMIND_BASE_URL": "http://127.0.0.1:9002/v1",
            "MINIMIND_MODEL": "minimind-3-ticket-v1",
            "MINIMIND_API_KEY": "sk-local",
            "MINIMIND_ENABLE_TRIAGE": "false",
            "MINIMIND_ENABLE_DRAFT_REPLY": "true",
        },
    )

    minimind_report = _load_minimind_report(project_root)

    rag = {
        "policy_hit_rate": _compare_metric(rule_report["baseline_policy_hit_rate"], rule_report["rag_policy_hit_rate"]),
        "history_hit_rate": _compare_metric(rule_report["baseline_history_hit_rate"], rule_report["rag_history_hit_rate"]),
        "kb_hit_rate": _compare_metric(rule_report["baseline_kb_hit_rate"], rule_report["rag_kb_hit_rate"]),
        "vector_retrieval_usage_rate": _round(rule_report["vector_retrieval_usage_rate"]),
        "citation_grounding_rate": _round(rule_report["citation_grounding_rate"]),
    }

    mcp = {
        "incident_delivery_success_rate": _compare_metric(
            mcp_disabled["incident_email_success_rate"],
            mcp_enabled["incident_email_success_rate"],
        ),
        "kb_candidate_success_rate": _compare_metric(
            mcp_disabled["kb_candidate_success_rate"],
            mcp_enabled["kb_candidate_success_rate"],
        ),
        "external_nonblocking_recovery_rate": _compare_metric(
            mcp_disabled["external_nonblocking_recovery_rate"],
            mcp_enabled["external_nonblocking_recovery_rate"],
        ),
        "enabled_trigger_rates": {
            "incident_email_trigger_rate": _round(mcp_enabled["incident_email_trigger_rate"]),
            "kb_candidate_trigger_rate": _round(mcp_enabled["kb_candidate_trigger_rate"]),
        },
        "enabled_avg_latency_ms": _round(mcp_enabled["external_delivery_avg_latency_ms"]),
    }

    lora_integration = {
        "reply_minimind_coverage_rate": _compare_metric(rule_report["reply_minimind_rate"], minimind_integration["reply_minimind_rate"]),
        "first_reply_usable_rate": _compare_metric(rule_report["first_reply_usable_rate"], minimind_integration["first_reply_usable_rate"]),
        "groundedness_rate": _compare_metric(rule_report["groundedness_rate"], minimind_integration["groundedness_rate"]),
        "policy_violation_rate": _compare_metric(rule_report["policy_violation_rate"], minimind_integration["policy_violation_rate"]),
        "hallucination_rate": _compare_metric(rule_report["hallucination_rate"], minimind_integration["hallucination_rate"]),
        "reply_rubric_proxy_mean": _compare_metric(rule_report["reply_rubric_proxy_mean"], minimind_integration["reply_rubric_proxy_mean"]),
        "fallback_case_rate": _compare_metric(rule_report["fallback_case_rate"], minimind_integration["fallback_case_rate"]),
        "classification_accuracy": _compare_metric(rule_report["classification_accuracy"], minimind_integration["classification_accuracy"]),
    }

    return {
        "methodology": {
            "principles": [
                "所有模块都采用同一批“公开 support ticket 语料中文化改写 + 本地业务侧表”样本进行对比。",
                "每次对比只改变一个模块，其余 workflow、规则和数据保持不变。",
                "对于基线为 0 的指标，只报告绝对提升，不虚构相对百分比。",
            ],
            "rag": "在相同 TicketFlow 主流程上，对比 hybrid RAG 与 keyword-only baseline 的证据命中率。",
            "mcp": "在相同 TicketFlow 主流程上，对比关闭外部协同与启用本地 SMTP sink 的外部通知成功率和非阻塞恢复率。",
            "lora": "在相同 MiniMind base 模型、相同 held-out 集、相同评分脚本上，对比微调前后的离线任务得分；同时在 TicketFlow 中验证 reply 节点是否真正切到本地微调模型。",
        },
        "agent_workflow": {
            "classification_accuracy": _round(rule_report["classification_accuracy"]),
            "category_macro_f1": _round(rule_report["category_macro_f1"]),
            "sla_risk_recall": _round(rule_report["sla_risk_recall"]),
            "action_type_accuracy": _round(rule_report["action_type_accuracy"]),
            "approval_required_recall": _round(rule_report["approval_required_recall"]),
            "target_status_accuracy": _round(rule_report["target_status_accuracy"]),
            "escalation_f1": _round(rule_report["escalation_f1"]),
            "approval_interception_rate": _round(rule_report["sensitive_action_approval_interception_rate"]),
            "first_reply_usable_rate": _round(rule_report["first_reply_usable_rate"]),
            "groundedness_rate": _round(rule_report["groundedness_rate"]),
            "policy_violation_rate": _round(rule_report["policy_violation_rate"]),
            "hallucination_rate": _round(rule_report["hallucination_rate"]),
            "reply_rubric_proxy_mean": _round(rule_report["reply_rubric_proxy_mean"]),
            "unsupported_sensitive_action_rate": _round(rule_report["unsupported_sensitive_action_rate"]),
        },
        "rag": rag,
        "mcp": mcp,
        "lora_offline": _build_lora_offline_section(minimind_report),
        "lora_in_ticketflow": lora_integration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate module contribution report for TicketFlow.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TicketFlow 项目根目录。",
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "module_contrib_report.json",
        help="输出 JSON 报告路径。",
    )
    args = parser.parse_args()

    report = build_report(args.project_root, sample_size=args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    localized = _localize_module_report(report)
    localized_json_path = args.output.with_name(args.output.stem + "_cn.json")
    localized_md_path = args.output.with_name(args.output.stem + "_cn.md")
    localized_json_path.write_text(json.dumps(localized, ensure_ascii=False, indent=2), encoding="utf-8")
    localized_md_path.write_text(_module_report_markdown(localized), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
