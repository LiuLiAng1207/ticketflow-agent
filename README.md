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
