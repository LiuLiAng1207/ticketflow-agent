# TicketFlow Agent Platform

TicketFlow 是一个面向客服与 IT 服务台场景的工单 Agent 平台。当前仓库保留稳定的 Streamlit 运维台，同时在 `production-claw-platform` 分支持续推进生产化：FastAPI 服务、Worker、任务状态、持久化审批、Outbox 幂等副作用和 Docker Compose 基础设施。

## 版本线

- `main`：稳定展示版。
- `interview-stable`：下周面试可讲的冻结基线。
- `production-claw-platform`：生产级平台改造分支，所有服务化升级都在这里推进。
- `interview-stable-2026-05-17`：冻结 tag，方便随时回退。

## 核心能力

- LangGraph 工单流程：分诊、检索、证据充分性判断、动作路由、工具级审批、动作执行、回复事实校验。
- FastAPI 服务：工单查询、异步任务、审批、审计、Outbox 和运营概览。
- Worker：Celery + Redis 模式执行工作流和 Outbox 投递，本地测试默认使用 eager 模式。
- Durable HITL：高风险动作审批请求会持久化，审批后可通过 API 恢复 workflow。
- Outbox Pattern：邮件、退款、升级等外部副作用先写 outbox，再由 worker 幂等消费。
- Streamlit 运维台：保留为内部工作台，展示模型、RAG、邮箱、任务队列、审批队列和 Outbox 状态。

## API

```bash
ticketflow-api
```

常用接口：

- `GET /healthz`
- `GET /readyz`
- `GET /api/v1/tickets`
- `GET /api/v1/tickets/{ticket_id}`
- `POST /api/v1/tickets/{ticket_id}/run`
- `POST /api/v1/tickets/{ticket_id}/run?sync=true`
- `GET /api/v1/tasks/{task_id}`
- `POST /api/v1/tasks/{task_id}/cancel`
- `GET /api/v1/approvals`
- `POST /api/v1/approvals/{approval_id}/decision`
- `GET /api/v1/outbox`
- `GET /api/v1/kg/health`
- `POST /api/v1/kg/tickets/{ticket_id}/rebuild`
- `GET /api/v1/kg/tickets/{ticket_id}`
- `GET /api/v1/kg/search`
- `GET /api/v1/skills`
- `GET /api/v1/skills/{skill_id}`
- `POST /api/v1/skills/reload`
- `POST /api/v1/skills/{skill_id}/enable`
- `POST /api/v1/skills/{skill_id}/disable`
- `POST /api/v1/skills/{skill_id}/run`
- `GET /api/v1/skills/runs`
- `GET /api/v1/claw/tasks`
- `GET /api/v1/claw/tasks/{task_id}`
- `POST /api/v1/claw/tasks/reload`
- `POST /api/v1/claw/tasks/{task_id}/run`
- `GET /api/v1/claw/runs/{run_id}`
- `GET /api/v1/claw/leaderboard`

默认异步运行会返回 `task_id`。如果工作流遇到退款或升级等敏感工具审批，任务会进入 `waiting_approval`，审批通过后 API 会自动 resume workflow 并更新任务状态。

## Worker

```bash
ticketflow-worker --mode local --once
ticketflow-worker --list-tasks
celery -A ticketflow.worker:celery_app worker --loglevel=INFO --pool=solo
```

当前注册任务：

- `run_ticket_workflow`
- `send_outbox_email`
- `run_claw_task`
- `build_knowledge_graph`
- `run_skill`

`build_knowledge_graph` rebuilds a ticket-level business graph from the current repository state. The graph is used for evidence-chain explanation, audit review, and future GraphRAG retrieval; it does not replace sufficiency checks or approval controls.

`run_skill` executes only allowlisted TicketFlow skills. Skills are registered from `skills/<skill_id>/skill.yaml` and `SKILL.md`, persisted through the repository backend, and audited through `skill_runs`. Skill Runtime is a governance layer, not an arbitrary code execution surface.

`run_claw_task` executes registered Claw benchmark tasks from `claw_tasks/*.yaml`. Each run records attempts, trajectory events, deterministic verifier scores, and leaderboard rows. Claw is an evaluation harness only; it does not bypass sufficiency checks, durable approvals, Skill permissions, or Outbox idempotency.

## Platform MCP

```bash
ticketflow-platform-mcp
```

TicketFlow Platform MCP exposes the production platform to external Agent clients through MCP tools, resources, and prompts. It is separate from the email MCP server. The email MCP server still handles SMTP delivery only; the platform MCP is the governed access layer for tickets, approvals, Outbox, KG, Skill Runtime, Claw, and the conversational Agent.

Default safety posture:

- `TICKETFLOW_MCP_TRANSPORT=stdio`
- `TICKETFLOW_MCP_HOST=127.0.0.1`
- `TICKETFLOW_MCP_PORT=8000`
- `TICKETFLOW_MCP_ALLOWED_SCOPES=read,eval`
- `TICKETFLOW_MCP_ENABLE_WRITE_TOOLS=false`
- `TICKETFLOW_MCP_ACTOR=mcp-client`

Read/eval tools are available by default. Write tools such as workflow submission, approval decisions, Skill execution, and conversational Agent actions require `TICKETFLOW_MCP_ENABLE_WRITE_TOOLS=true`. Even when enabled, these tools call existing governed paths and do not bypass sufficiency checks, durable approvals, Skill permissions, Claw Runtime, or Outbox idempotency.

Core MCP tools:

- `ticketflow.tickets.list`
- `ticketflow.tickets.get`
- `ticketflow.tickets.audit`
- `ticketflow.workflow.run`
- `ticketflow.approvals.list`
- `ticketflow.approvals.decide`
- `ticketflow.outbox.list`
- `ticketflow.kg.explain_ticket`
- `ticketflow.kg.search`
- `ticketflow.skills.list`
- `ticketflow.skills.run`
- `ticketflow.claw.tasks.list`
- `ticketflow.claw.run`
- `ticketflow.claw.runs.get`
- `ticketflow.claw.leaderboard`
- `ticketflow.agent.chat`

Core MCP resources:

- `ticketflow://tickets/{ticket_id}`
- `ticketflow://tickets/{ticket_id}/audit`
- `ticketflow://kg/tickets/{ticket_id}`
- `ticketflow://skills/{skill_id}`
- `ticketflow://claw/runs/{run_id}`

Core MCP prompts:

- `explain_ticket(ticket_id)`
- `triage_ticket(title, body, customer_tier, product)`
- `kb_candidate_from_ticket(ticket_id)`
- `claw_failure_analysis(run_id)`

## Docker Compose

```bash
docker compose config
docker compose up api worker postgres redis qdrant neo4j minio
```

Compose 包含：

- `api`
- `worker`
- `postgres`
- `redis`
- `qdrant`
- `neo4j`
- `minio`

当前业务默认仍使用 SQLite fallback，PostgreSQL/Qdrant/Neo4j/MinIO 先作为可启动基础设施，后续按阶段切换正式读写路径。

## 本地运行

```bash
python -m pip install -e .[dev]
ticketflow-reset
ticketflow-app
```

或者直接启动 Streamlit：

```bash
streamlit run src/ticketflow/app.py
```

## 配置

复制 `.env.example` 为 `.env` 后按需修改。本地密钥、SMTP 授权码、API Key、数据库和缓存文件不要提交到 Git。

常用生产骨架配置：

```env
APP_ENV=development
DATABASE_BACKEND=sqlite
DATABASE_URL=postgresql://ticketflow:ticketflow@localhost:5432/ticketflow
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_TASK_ALWAYS_EAGER=true
ENABLE_PRODUCTION_SERVICES=false
SKILL_REGISTRY_BACKEND=repository
SKILLS_DIR=skills
ENABLE_SKILL_RUNTIME=true
CLAW_TASKS_DIR=claw_tasks
```

Docker Compose 中 `CELERY_TASK_ALWAYS_EAGER=false`，worker 会通过 Redis broker 消费任务。

## 测试

```bash
python -m pytest -q
python -m pytest tests/test_api.py tests/test_worker.py tests/test_production_persistence.py -q
docker compose config
```

如果 Docker Desktop 未启动，只能验证 `docker compose config`，不能实际 `up` 服务。

## 下一阶段

1. PostgreSQL repository 深度替换 SQLite fallback。
2. Streamlit 运维台全面切 API 任务状态。
3. Claw Harness、Skill Runtime、Knowledge Graph 和 Observability。
4. Locust / k6 / worker 并发压测与稳定性报告。
