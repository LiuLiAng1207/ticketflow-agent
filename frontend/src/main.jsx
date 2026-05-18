import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_TICKETFLOW_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

const statusLabels = {
  open: "开放",
  waiting_on_customer: "等待客户补充",
  pending_human: "待人工处理",
  pending_finance: "待财务审批",
  resolved: "已解决",
};

const intentLabels = {
  identity: "身份说明",
  ops_summary: "运营统计",
  query_ticket: "查询工单",
  run_ticket: "启动工作流",
  list_approvals: "审批队列",
  list_outbox: "Outbox 状态",
  explain_ticket_graph: "证据链解释",
  submit_kb_candidate: "知识沉淀",
  create_ticket: "创建工单",
  freeform_answer: "云端解释",
  help: "帮助",
  refuse_unsafe_action: "安全拒绝",
};

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload?.detail?.message || payload?.detail || payload?.error || response.statusText);
  }
  return payload;
}

function Pill({ children, tone = "neutral" }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function EmptyState({ title, text }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
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
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "我是 TicketFlow 工单协同 Agent。你可以问我“你是谁”、统计待处理工单、查询或处理某个工单，也可以查看审批、Outbox、Claw 评测和证据链。",
      intent: "identity",
      model_source: "deterministic",
      events: [{ event_type: "welcome", title: "进入工作台", status: "completed", summary: "对话线程已就绪。" }],
    },
  ]);
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

  async function loadAll() {
    setLastError("");
    try {
      const [healthz, readyz, ticketData, ops, taskData, approvalData, outboxData, clawData, boardData] = await Promise.all([
        api("/healthz"),
        api("/readyz"),
        api("/api/v1/tickets?limit=80"),
        api("/api/v1/ops/summary"),
        api("/api/v1/tasks?limit=20").catch(() => ({ tasks: [] })),
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

  useEffect(() => {
    loadAll();
    const timer = window.setInterval(loadAll, 12000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function sendMessage(messageText = input) {
    const text = messageText.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMessages((items) => [...items, { role: "user", content: text }]);
    try {
      const result = await api("/api/v1/agent/chat", {
        method: "POST",
        body: JSON.stringify({ message: text, actor: "web-console" }),
      });
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: result.reply,
          intent: result.intent,
          model_source: result.model_source,
          data: result.data,
          actions: result.actions || [],
          events: result.events || [],
        },
      ]);
      loadAll();
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

  async function runClawTask(taskId) {
    setBusy(true);
    try {
      const result = await api(`/api/v1/claw/tasks/${taskId}/run?sync=true&pass_k=1`, {
        method: "POST",
        body: JSON.stringify({ actor: "web-console", config: {} }),
      });
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: `Claw 任务 ${taskId} 已完成，平均分 ${Number(result.result?.average_score || 0).toFixed(3)}。`,
          intent: "claw_run",
          model_source: "deterministic",
          data: result,
          events: [{ event_type: "claw_run", title: "运行 Claw 任务", status: "completed", summary: taskId }],
        },
      ]);
      loadAll();
    } catch (error) {
      setLastError(String(error.message || error));
    } finally {
      setBusy(false);
    }
  }

  const quickPrompts = [
    "你是谁",
    "帮我分析待处理工单的数量",
    selectedTicket ? `查询 ${selectedTicket.ticket_id} 的状态` : "",
    selectedTicket ? `解释 ${selectedTicket.ticket_id} 的证据链` : "",
    "查看待审批",
    "查看 Outbox 投递状态",
  ].filter(Boolean);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">TF</span>
          <div>
            <h1>TicketFlow</h1>
            <p>Agent 工单指挥台</p>
          </div>
        </div>

        <section className="status-block">
          <div className="status-row">
            <span>API 服务</span>
            <Pill tone={health.api === "可用" ? "green" : "red"}>{health.api}</Pill>
          </div>
          <div className="status-row">
            <span>模型后端</span>
            <Pill tone={health.ready?.llm?.enabled ? "blue" : "amber"}>
              {health.ready?.llm?.enabled ? health.ready?.llm?.model || "已配置" : "未配置"}
            </Pill>
          </div>
          <div className="status-row">
            <span>运行模式</span>
            <Pill>{health.ready?.database?.backend || "检测中"}</Pill>
          </div>
        </section>

        <section className="summary-grid">
          <Metric label="开放工单" value={summary?.open_tickets ?? "-"} />
          <Metric label="企业客户" value={summary?.enterprise_tickets ?? "-"} />
          <Metric label="风险关注" value={summary?.risk_watch_tickets ?? "-"} />
          <Metric label="退款相关" value={summary?.refund_related_tickets ?? "-"} />
        </section>

        <div className="search-box">
          <span>工单搜索</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="编号、标题、产品、类别" />
        </div>

        <section className="ticket-list">
          {filteredTickets.map((ticket) => (
            <button
              key={ticket.ticket_id}
              className={`ticket-item ${selectedTicket?.ticket_id === ticket.ticket_id ? "active" : ""}`}
              onClick={() => setSelectedTicketId(ticket.ticket_id)}
            >
              <span>{ticket.ticket_id}</span>
              <strong>{ticket.title}</strong>
              <small>
                {ticket.product} · {ticket.customer_tier}
              </small>
            </button>
          ))}
        </section>
      </aside>

      <section className="conversation">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">Agent Conversation</p>
            <h2>可对话、可追踪、可审批的工单 Agent</h2>
          </div>
          <button className="ghost-button" onClick={loadAll}>
            刷新状态
          </button>
        </header>

        {lastError ? <div className="error-banner">{lastError}</div> : null}

        <div className="quick-prompts">
          {quickPrompts.map((prompt) => (
            <button key={prompt} onClick={() => sendMessage(prompt)} disabled={busy}>
              {prompt}
            </button>
          ))}
        </div>

        <ChatThread messages={messages} />

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            sendMessage();
          }}
        >
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="例如：帮我分析待处理工单的数量，或者：处理 TCK-0001"
            rows={3}
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "处理中" : "发送"}
          </button>
        </form>
      </section>

      <aside className="inspector">
        <section className="panel hero-panel">
          <p className="eyebrow">Selected Ticket</p>
          {selectedTicket ? (
            <>
              <h3>{selectedTicket.ticket_id}</h3>
              <p>{selectedTicket.title}</p>
              <div className="pill-row">
                <Pill tone="blue">{selectedTicket.expected_category || "未分类"}</Pill>
                <Pill tone="green">{statusLabels[selectedTicket.status] || selectedTicket.status}</Pill>
                <Pill>{selectedTicket.customer_tier}</Pill>
              </div>
              <div className="detail-line">
                <span>产品</span>
                <strong>{selectedTicket.product}</strong>
              </div>
              <div className="detail-line">
                <span>订单</span>
                <strong>{selectedTicket.linked_order_id || "无"}</strong>
              </div>
            </>
          ) : (
            <EmptyState title="暂无工单" text="API 返回后会显示选中的工单详情。" />
          )}
        </section>

        <QueuePanel title="任务队列" items={tasks} empty="暂无工作流任务" idKey="task_id" />
        <QueuePanel title="审批队列" items={approvals} empty="暂无待审批请求" idKey="approval_id" />
        <QueuePanel title="Outbox 投递" items={outbox} empty="暂无 Outbox 事件" idKey="event_id" />

        <section className="panel">
          <div className="panel-title">
            <span>Claw 评测中心</span>
            <Pill tone="blue">{leaderboard.length} 条榜单</Pill>
          </div>
          {clawTasks.length ? (
            clawTasks.slice(0, 5).map((task) => (
              <div className="claw-row" key={task.task_id}>
                <div>
                  <strong>{task.name || task.task_id}</strong>
                  <span>{task.goal}</span>
                </div>
                <button onClick={() => runClawTask(task.task_id)} disabled={busy}>
                  运行
                </button>
              </div>
            ))
          ) : (
            <EmptyState title="Claw 未加载" text="点击刷新或调用 Claw reload 后显示任务。" />
          )}
          {leaderboard.slice(0, 4).map((row) => (
            <div className="leaderboard-row" key={`${row.task_id}-${row.latest_run_id}`}>
              <span>{row.task_id}</span>
              <strong>{Number(row.best_score || 0).toFixed(3)}</strong>
            </div>
          ))}
        </section>
      </aside>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChatThread({ messages }) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);
  return (
    <section className="chat-thread">
      {messages.map((message, index) => (
        <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
          <div className="avatar">{message.role === "user" ? "你" : "AI"}</div>
          <div className="message-body">
            <p>{message.content}</p>
            {message.role === "assistant" ? (
              <div className="message-meta">
                <Pill tone="blue">{intentLabels[message.intent] || message.intent || "回复"}</Pill>
                <Pill tone={message.model_source === "deepseek" ? "green" : "neutral"}>{message.model_source || "deterministic"}</Pill>
              </div>
            ) : null}
            {message.events?.length ? <EventTimeline events={message.events} /> : null}
            {message.actions?.length ? (
              <div className="action-strip">
                {message.actions.map((action, actionIndex) => (
                  <span key={actionIndex}>{action.type || action.method || "action"}</span>
                ))}
              </div>
            ) : null}
            {message.data && Object.keys(message.data).length ? (
              <details>
                <summary>查看结构化结果</summary>
                <pre>{JSON.stringify(message.data, null, 2)}</pre>
              </details>
            ) : null}
          </div>
        </article>
      ))}
      <div ref={endRef} />
    </section>
  );
}

function EventTimeline({ events }) {
  return (
    <div className="timeline">
      {events.map((event, index) => (
        <div className={`timeline-event ${event.status || "completed"}`} key={`${event.event_type}-${index}`}>
          <span />
          <div>
            <strong>{event.title || event.event_type}</strong>
            <small>{event.summary}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

function QueuePanel({ title, items, empty, idKey }) {
  return (
    <section className="panel compact">
      <div className="panel-title">
        <span>{title}</span>
        <Pill>{items.length}</Pill>
      </div>
      {items.length ? (
        items.slice(0, 4).map((item) => (
          <div className="queue-row" key={item[idKey] || item.task_id || item.event_id}>
            <strong>{item[idKey] || item.ticket_id}</strong>
            <span>{item.status || item.operation_type || item.tool_name || "running"}</span>
          </div>
        ))
      ) : (
        <EmptyState title={empty} text="队列为空时说明当前没有阻塞项。" />
      )}
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
