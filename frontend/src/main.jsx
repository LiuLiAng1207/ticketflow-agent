import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_TICKETFLOW_API_BASE || "http://127.0.0.1:8000").trim().replace(/\/$/, "");

const statusLabels = {
  open: "开放",
  queued: "已排队",
  running: "执行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  waiting_approval: "等待审批",
  pending: "待处理",
};

const categoryLabels = {
  billing_refund: "退款账单",
  technical_issue: "技术故障",
  delivery_issue: "交付物流",
  account_access: "账号访问",
  general_inquiry: "一般咨询",
};

const intentLabels = {
  identity: "身份说明",
  ops_summary: "运营统计",
  query_ticket: "查询工单",
  run_ticket: "启动流程",
  batch_status: "批量结果",
  list_approvals: "审批队列",
  list_outbox: "Outbox",
  explain_ticket_graph: "证据链解释",
  submit_kb_candidate: "知识沉淀",
  create_ticket: "创建工单",
  run_skill: "运行 Skill",
  claw_run: "Claw 运行",
  freeform_answer: "云端解释",
  help: "帮助",
  refuse_unsafe_action: "安全拒绝",
  error: "异常",
};

const modelSourceLabels = {
  deterministic: "治理路由",
  deepseek: "DeepSeek",
  fallback: "保守兜底",
};

const actionLabels = {
  run_skill: "调用 Skill",
  list_tasks: "查看任务",
  list_tickets: "查看工单",
  list_approvals: "查看审批",
  list_outbox: "查看 Outbox",
  get_ticket: "查看工单详情",
  get_task: "查看任务",
  get_ticket_graph: "查看证据链",
  create_outbox: "写入 Outbox",
  open_approvals: "打开审批",
};

const runwaySteps = ["识别意图", "读取上下文", "调用 Skill", "进入队列", "审计留痕"];

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || payload.message || message;
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json();
}

function shortId(value, head = 9, tail = 5) {
  if (!value) return "无编号";
  const text = String(value);
  if (text.length <= head + tail + 1) return text;
  return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

function App() {
  const [health, setHealth] = useState({ api: "检测中", ready: null });
  const [tickets, setTickets] = useState([]);
  const [selectedTicketId, setSelectedTicketId] = useState("");
  const [summary, setSummary] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [approvals, setApprovals] = useState([]);
  const [outbox, setOutbox] = useState([]);
  const [clawTasks, setClawTasks] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);
  const [auditEvents, setAuditEvents] = useState([]);
  const [ticketGraph, setTicketGraph] = useState(null);
  const [inspectorBusy, setInspectorBusy] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "我是 TicketFlow Agent。你可以让我查询工单、批量处理低风险工单、解释证据链、查看审批和 Outbox。",
      intent: "identity",
      model_source: "deterministic",
      events: [{ event_type: "boot", title: "Agent 已就绪", status: "completed", summary: "前端已连接 TicketFlow API。" }],
    },
  ]);
  const [agentMemory, setAgentMemory] = useState({ lastBatchTaskIds: [], lastBatchRequestedCount: 0 });
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [lastError, setLastError] = useState("");

  const selectedTicket = useMemo(
    () => tickets.find((ticket) => ticket.ticket_id === selectedTicketId) || tickets[0],
    [tickets, selectedTicketId],
  );

  const filteredTickets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return tickets;
    return tickets.filter((ticket) =>
      [ticket.ticket_id, ticket.title, ticket.product, ticket.customer_tier, ticket.expected_category]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [tickets, query]);

  const latest = latestAssistant(messages);
  const latestEvents = latest?.events || [];
  const batchRows = latest?.data?.tasks || latest?.data?.skill_run?.result?.tasks || [];
  const activeTaskCount = tasks.filter((task) => ["queued", "running", "waiting_approval"].includes(task.status)).length;
  const pendingApprovalCount = approvals.filter((item) => (item.status || "pending") === "pending").length;

  async function loadAll() {
    setLastError("");
    try {
      const [healthz, readyz, ticketData, ops, taskData, approvalData, outboxData, clawData, boardData] = await Promise.all([
        api("/healthz"),
        api("/readyz"),
        api("/api/v1/tickets?limit=80"),
        api("/api/v1/ops/summary"),
        api("/api/v1/tasks?limit=30").catch(() => ({ tasks: [] })),
        api("/api/v1/approvals?limit=20").catch(() => ({ approvals: [] })),
        api("/api/v1/outbox?limit=20").catch(() => ({ events: [] })),
        api("/api/v1/claw/tasks?limit=20").catch(() => ({ tasks: [] })),
        api("/api/v1/claw/leaderboard?limit=20").catch(() => ({ leaderboard: [] })),
      ]);
      setHealth({ api: healthz.status === "ok" ? "可用" : "异常", ready: readyz });
      setTickets(ticketData.tickets || []);
      setSummary(ops);
      setTasks(taskData.tasks || []);
      setApprovals(approvalData.approvals || []);
      setOutbox(outboxData.events || []);
      setClawTasks(clawData.tasks || []);
      setLeaderboard(boardData.leaderboard || []);
      if (!selectedTicketId && ticketData.tickets?.length) {
        setSelectedTicketId(ticketData.tickets[0].ticket_id);
      }
    } catch (error) {
      setHealth({ api: "不可用", ready: null });
      setLastError(String(error.message || error));
    }
  }

  async function loadTicketContext(ticketId) {
    if (!ticketId) return;
    setInspectorBusy(true);
    try {
      const [audit, graph] = await Promise.all([
        api(`/api/v1/tickets/${ticketId}/audit`).catch(() => ({ events: [] })),
        api(`/api/v1/kg/tickets/${ticketId}`)
          .then(async (currentGraph) => {
            if (currentGraph?.enabled && !currentGraph?.node_count) {
              await api(`/api/v1/kg/tickets/${ticketId}/rebuild`, { method: "POST" }).catch(() => null);
              return api(`/api/v1/kg/tickets/${ticketId}`).catch(() => currentGraph);
            }
            return currentGraph;
          })
          .catch(() => ({ enabled: false, nodes: [], edges: [], node_count: 0, edge_count: 0 })),
      ]);
      setAuditEvents(audit.events || []);
      setTicketGraph(graph);
    } finally {
      setInspectorBusy(false);
    }
  }

  useEffect(() => {
    loadAll();
    const timer = window.setInterval(loadAll, 12000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (selectedTicket?.ticket_id) {
      loadTicketContext(selectedTicket.ticket_id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTicket?.ticket_id]);

  async function sendMessage(messageText = input) {
    const text = messageText.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMessages((items) => [...items, { role: "user", content: text }]);
    try {
      const result = await api("/api/v1/agent/chat", {
        method: "POST",
        body: JSON.stringify({
          message: text,
          actor: "web-console",
          client_context: {
            selected_ticket_id: selectedTicket?.ticket_id,
            last_batch_task_ids: agentMemory.lastBatchTaskIds,
            last_batch_requested_count: agentMemory.lastBatchRequestedCount,
          },
        }),
      });
      const assistantMessage = {
        role: "assistant",
        content: result.reply,
        intent: result.intent,
        model_source: result.model_source,
        data: result.data,
        actions: result.actions || [],
        events: result.events || [],
      };
      setMessages((items) => [...items, assistantMessage]);

      const batchResult = result.data?.skill_run?.result;
      if (result.intent === "run_skill" && Array.isArray(batchResult?.tasks)) {
        setAgentMemory({
          lastBatchTaskIds: batchResult.tasks.map((task) => task.task_id).filter(Boolean),
          lastBatchRequestedCount: batchResult.requested_count || batchResult.count || batchResult.tasks.length,
        });
      }
      if (result.intent === "batch_status" && Array.isArray(result.data?.tasks)) {
        setAgentMemory((current) => ({
          ...current,
          lastBatchTaskIds: result.data.tasks.map((task) => task.task_id).filter(Boolean),
        }));
      }

      await loadAll();
      if (selectedTicket?.ticket_id) {
        await loadTicketContext(selectedTicket.ticket_id);
      }
    } catch (error) {
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: `Agent API 暂不可用：${error.message || error}`,
          intent: "error",
          model_source: "fallback",
          events: [{ event_type: "error", title: "请求失败", status: "failed", summary: String(error.message || error) }],
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function reloadClawTasks() {
    setBusy(true);
    try {
      await api("/api/v1/claw/tasks/reload", { method: "POST" });
      await loadAll();
    } catch (error) {
      setLastError(String(error.message || error));
    } finally {
      setBusy(false);
    }
  }

  async function runClawTask(taskId) {
    setBusy(true);
    try {
      const result = await api(`/api/v1/claw/tasks/${taskId}/run?sync=true&pass_k=1`, { method: "POST" });
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: `Claw 任务 ${taskId} 已完成，平均分 ${Number(result.run?.summary?.average_score || result.result?.average_score || 0).toFixed(3)}。`,
          intent: "claw_run",
          model_source: "deterministic",
          data: result,
          events: [{ event_type: "claw_run", title: "运行 Claw 评测", status: "completed", summary: taskId }],
        },
      ]);
      await loadAll();
    } catch (error) {
      setLastError(String(error.message || error));
    } finally {
      setBusy(false);
    }
  }

  const quickPrompts = [
    "你是谁",
    "帮我分析待处理工单的数量",
    "帮我批量处理10张工单",
    "你处理了哪10条工单，处理结果分别是什么",
    selectedTicket ? `查询 ${selectedTicket.ticket_id} 的状态` : "",
    selectedTicket ? `解释 ${selectedTicket.ticket_id} 的证据链` : "",
    "查看待审批",
  ].filter(Boolean);

  return (
    <div className="ops-shell">
      <aside className="ticket-radar">
        <div className="brand-lockup">
          <div className="brand-orb">TF</div>
          <div>
            <p>TicketFlow</p>
            <strong>Agent Command</strong>
          </div>
        </div>

        <div className="health-stack">
          <StatusDot label="API" value={health.api} ok={health.api === "可用"} />
          <StatusDot label="模型" value={health.ready?.model_backend || "读取中"} />
          <StatusDot label="图谱" value={health.ready?.kg_backend || "未配置"} />
        </div>

        <div className="radar-metrics">
          <Metric label="开放" value={summary?.open_tickets ?? "—"} />
          <Metric label="风险" value={summary?.risk_watch_tickets ?? "—"} />
          <Metric label="审批" value={pendingApprovalCount} />
          <Metric label="任务" value={activeTaskCount} />
        </div>

        <div className="rail-head">
          <div>
            <span>工单雷达</span>
            <small>{filteredTickets.length} 条可见</small>
          </div>
          <button onClick={loadAll}>同步</button>
        </div>
        <input className="radar-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索编号、标题、产品或等级" />
        <div className="ticket-stream">
          {filteredTickets.map((ticket) => (
            <button
              className={`ticket-row ${selectedTicket?.ticket_id === ticket.ticket_id ? "active" : ""}`}
              key={ticket.ticket_id}
              onClick={() => setSelectedTicketId(ticket.ticket_id)}
            >
              <span>{ticket.ticket_id}</span>
              <strong>{ticket.title}</strong>
              <small>
                {categoryLabels[ticket.expected_category] || ticket.expected_category} / {ticket.customer_tier}
              </small>
            </button>
          ))}
        </div>
      </aside>

      <main className="command-canvas">
        <section className="command-hero">
          <div>
            <p className="eyebrow">Agent Operations</p>
            <h1>工单处理，不再像黑盒。</h1>
            <span>每次对话都会显示意图、工具调用、任务队列、证据链和治理边界。</span>
          </div>
          <div className="hero-actions">
            <button onClick={() => sendMessage("帮我批量处理10张工单")} disabled={busy}>
              批量处理低风险工单
            </button>
            <button className="secondary" onClick={() => sendMessage(selectedTicket ? `解释 ${selectedTicket.ticket_id} 的证据链` : "解释当前工单的证据链")} disabled={busy}>
              解释证据链
            </button>
          </div>
        </section>

        <section className="agent-runway">
          {runwaySteps.map((step, index) => (
            <div className={index <= Math.max(0, latestEvents.length - 1) ? "runway-step active" : "runway-step"} key={step}>
              <i>{index + 1}</i>
              <span>{step}</span>
            </div>
          ))}
        </section>

        {lastError ? <div className="error-banner">{lastError}</div> : null}

        <div className="quick-prompts">
          {quickPrompts.map((prompt) => (
            <button key={prompt} onClick={() => sendMessage(prompt)} disabled={busy}>
              {prompt}
            </button>
          ))}
        </div>

        <section className="dialogue-surface">
          <ChatThread messages={messages} />
          <BatchResultInline rows={batchRows} />
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              sendMessage();
            }}
          >
            <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="问 Agent：帮我批量处理10张工单，然后问处理了哪10条" />
            <button disabled={busy || !input.trim()}>{busy ? "执行中" : "发送"}</button>
          </form>
        </section>
      </main>

      <aside className="governance-inspector">
        <section className="selected-ticket">
          <div className="section-kicker">当前工单</div>
          {selectedTicket ? (
            <>
              <h2>{selectedTicket.ticket_id}</h2>
              <p>{selectedTicket.title}</p>
              <div className="pill-row">
                <Pill tone="blue">{categoryLabels[selectedTicket.expected_category] || selectedTicket.expected_category}</Pill>
                <Pill tone={selectedTicket.customer_tier === "enterprise" ? "amber" : "green"}>{selectedTicket.customer_tier}</Pill>
                <Pill>{statusLabels[selectedTicket.status] || selectedTicket.status}</Pill>
              </div>
            </>
          ) : (
            <EmptyState title="未选择工单" text="从左侧选择一个工单后显示证据链和治理链。" />
          )}
        </section>

        <section className="inspector-panel evidence-panel">
          <PanelTitle title="证据链与治理链" value={inspectorBusy ? "加载中" : "已同步"} tone={inspectorBusy ? "amber" : "green"} />
          <EvidenceGraph graph={ticketGraph} />
          <AuditTimeline events={auditEvents} />
        </section>

        <section className="inspector-panel">
          <PanelTitle title="最近 Agent 轨迹" value={`${latestEvents.length || 0} 步`} tone="blue" />
          <EventTimeline events={latestEvents} />
        </section>

        <QueuePanel title="任务队列" items={tasks} empty="暂无工作流任务" type="task" />
        <QueuePanel title="审批队列" items={approvals} empty="暂无待审批请求" type="approval" />
        <QueuePanel title="Outbox 投递" items={outbox} empty="暂无 Outbox 事件" type="outbox" />
        <ClawPanel tasks={clawTasks} leaderboard={leaderboard} busy={busy} onReload={reloadClawTasks} onRun={runClawTask} />
      </aside>
    </div>
  );
}

function latestAssistant(messages) {
  return [...messages].reverse().find((message) => message.role === "assistant");
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusDot({ label, value, ok = true }) {
  return (
    <div className="status-dot">
      <i className={ok ? "ok" : "warn"} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChatThread({ messages }) {
  const ref = useRef(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [messages]);
  return (
    <div className="chat-thread" ref={ref}>
      {messages.map((message, index) => (
        <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
          <div className="avatar">{message.role === "assistant" ? "AI" : "你"}</div>
          <div className="bubble">
            <p>{message.content}</p>
            {message.role === "assistant" ? (
              <div className="message-meta">
                <Pill tone="blue">{intentLabels[message.intent] || message.intent || "回复"}</Pill>
                <Pill tone={message.model_source === "deepseek" ? "green" : "neutral"}>
                  {modelSourceLabels[message.model_source] || message.model_source || "治理路由"}
                </Pill>
              </div>
            ) : null}
            {message.events?.length ? <EventTimeline events={message.events} compact /> : null}
            {message.actions?.length ? (
              <div className="action-strip">
                {message.actions.map((action, actionIndex) => (
                  <span key={actionIndex}>{actionLabels[action.type] || action.type || action.method || "动作"}</span>
                ))}
              </div>
            ) : null}
            {message.data ? (
              <details className="json-details">
                <summary>查看结构化结果</summary>
                <pre>{JSON.stringify(message.data, null, 2)}</pre>
              </details>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function BatchResultInline({ rows }) {
  if (!rows?.length) return null;
  return (
    <section className="batch-strip">
      <PanelTitle title="批量处理结果" value={`${rows.length} 条`} tone="blue" />
      <div className="batch-grid">
        {rows.slice(0, 10).map((task, index) => (
          <div className="batch-card" key={task.task_id || index}>
            <span>{task.ticket_id}</span>
            <strong>{statusLabels[task.status] || task.status_label || task.status || "已创建"}</strong>
            <small>{shortId(task.task_id)}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function EventTimeline({ events, compact = false }) {
  if (!events?.length) {
    return <EmptyState title="暂无轨迹" text="Agent 执行后会显示意图识别、工具调用、审批和完成状态。" />;
  }
  return (
    <div className={`timeline ${compact ? "compact-timeline" : ""}`}>
      {events.map((event, index) => (
        <div className="timeline-row" key={`${event.event_type}-${index}`}>
          <span className={`timeline-dot ${event.status || "completed"}`} />
          <div>
            <strong>{event.title || event.event_type}</strong>
            {event.summary ? <p>{event.summary}</p> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function EvidenceGraph({ graph }) {
  if (!graph) {
    return <EmptyState title="正在读取证据链" text="选择工单后会自动读取图谱、审计和治理事件。" />;
  }
  if (!graph.enabled) {
    return <EmptyState title="知识图谱未启用" text="启动 API 时设置 KG_BACKEND=memory 或 neo4j 后即可展示证据链。" />;
  }
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const evidenceNodes = nodes.filter((node) => ["Policy", "KBArticle", "HistoryCase", "Order", "Customer", "Ticket", "WorkflowTask"].includes(node.label));
  return (
    <div className="graph-box">
      <div className="graph-stats">
        <Metric label="业务节点" value={graph.node_count ?? nodes.length} />
        <Metric label="关系边" value={graph.edge_count ?? edges.length} />
      </div>
      <div className="node-list">
        {evidenceNodes.slice(0, 6).map((node) => (
          <div className="node-row" key={node.id}>
            <span>{node.label}</span>
            <strong>{node.properties?.title || node.properties?.name || node.id}</strong>
          </div>
        ))}
      </div>
      <p className="subhead">关键关系</p>
      <div className="edge-list">
        {edges.slice(0, 6).map((edge, index) => (
          <span key={`${edge.source}-${edge.target}-${index}`}>
            {edge.source} → {edge.type} → {edge.target}
          </span>
        ))}
      </div>
    </div>
  );
}

function AuditTimeline({ events }) {
  const important = (events || []).filter((event) =>
    [
      "sufficiency_assessed",
      "action_routed",
      "tool_approval_requested",
      "tool_approval_completed",
      "reply_fact_checked",
      "reply_rewritten_or_downgraded",
      "execute_action",
      "ticket_created_from_chat",
    ].includes(event.event_type),
  );
  return (
    <div className="audit-list">
      <p className="subhead">审计事件</p>
      {important.length ? (
        important.slice(0, 6).map((event, index) => (
          <div className="audit-row" key={`${event.event_type}-${index}`}>
            <span>{event.event_type}</span>
            <strong>{event.detail || event.actor || "已记录"}</strong>
          </div>
        ))
      ) : (
        <EmptyState title="暂无审计事件" text="工单进入工作流后会记录检索、路由、审批和回复校验。" />
      )}
    </div>
  );
}

function QueuePanel({ title, items, empty, type }) {
  return (
    <section className="inspector-panel compact">
      <PanelTitle title={title} value={String(items.length)} />
      {items.length ? (
        items.slice(0, 5).map((item) => {
          const view = queueView(item, type);
          return (
            <div className="queue-row" key={view.key}>
              <div>
                <strong>{view.title}</strong>
                <small>{view.subtitle}</small>
              </div>
              <span>{view.status}</span>
            </div>
          );
        })
      ) : (
        <EmptyState title={empty} text="队列为空说明当前没有阻塞项。" />
      )}
    </section>
  );
}

function ClawPanel({ tasks, leaderboard, busy, onReload, onRun }) {
  return (
    <section className="inspector-panel compact">
      <PanelTitle title="Claw 评测中心" value={String(tasks.length)} tone="blue" />
      {tasks.length ? (
        tasks.slice(0, 5).map((task) => (
          <div className="claw-row" key={task.task_id}>
            <div>
              <strong>{task.name || task.task_id}</strong>
              <span>{task.goal || "Agent 任务评测"}</span>
            </div>
            <button onClick={() => onRun(task.task_id)} disabled={busy}>
              运行
            </button>
          </div>
        ))
      ) : (
        <div className="stack-gap">
          <EmptyState title="Claw 未加载" text="点击加载任务后显示评测集、运行按钮、分数和失败原因。" />
          <button className="full-button" onClick={onReload} disabled={busy}>
            加载 Claw 任务
          </button>
        </div>
      )}
      {leaderboard.slice(0, 4).map((row) => (
        <div className="leaderboard-row" key={`${row.task_id}-${row.latest_run_id}`}>
          <span>{row.task_id}</span>
          <strong>{Number(row.average_score || 0).toFixed(3)}</strong>
        </div>
      ))}
    </section>
  );
}

function queueView(item, type) {
  if (type === "task") {
    return {
      key: item.task_id,
      title: item.ticket_id || "未知工单",
      subtitle: `任务 ${shortId(item.task_id)} / ${item.mode || "async"}`,
      status: statusLabels[item.status] || item.status || "未知",
    };
  }
  if (type === "approval") {
    return {
      key: item.approval_id,
      title: item.ticket_id || item.tool_name || "审批请求",
      subtitle: `审批 ${shortId(item.approval_id)} / ${item.tool_name || "工具"}`,
      status: statusLabels[item.status] || item.status || "待审批",
    };
  }
  return {
    key: item.event_id,
    title: item.ticket_id || item.operation_type || "Outbox",
    subtitle: `${item.operation_type || "外部投递"} / ${shortId(item.event_id)}`,
    status: statusLabels[item.status] || item.status || "未知",
  };
}

function PanelTitle({ title, value, tone = "neutral" }) {
  return (
    <div className="panel-title">
      <span>{title}</span>
      <Pill tone={tone}>{value}</Pill>
    </div>
  );
}

function Pill({ children, tone = "neutral" }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

function EmptyState({ title, text }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
