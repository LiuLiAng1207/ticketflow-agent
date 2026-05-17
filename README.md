# TicketFlow

TicketFlow 是一个面向客服中心与 IT 服务台场景的工单协同智能体平台。项目目标不是做一个单轮客服聊天机器人，而是持续演进一个可运行、可评测、可审计、可恢复、可接入外部工具的业务型 Agent 系统。

当前稳定版仍保留 Streamlit 运维台和 SQLite 本地运行方式；`production-claw-platform` 分支开始把系统升级为生产服务骨架，为后续 PostgreSQL、Celery、Qdrant、Neo4j、MinIO、Skill Runtime、Claw Harness 和观测系统铺路。

## 版本线

- `main`：当前稳定主线。
- `interview-stable`：已冻结的稳定演示基线。
- `interview-stable-2026-05-17`：对应稳定演示基线的 Git tag。
- `production-claw-platform`：生产级平台改造分支，所有服务化升级在这里推进。

## 当前能力

- 基于 LangGraph 的状态化工单工作流。
- 工单分诊、混合检索、证据充分性判断、动作路由、工具级审批、动作执行、回复生成与事实校验。
- 多来源证据：知识库、策略规则、历史工单、订单、客户资料与附件证据。
- 本地 MiniMind 双 LoRA 与云端 OpenAI-compatible 模型后端切换。
- MCP 邮件工具服务，支持真实 SMTP 链路的外部协同。
- Streamlit 内部运维台，支持模型控制、工单运行、审批和链路展示。
- 离线评测脚本和报告，用于追踪 RAG、动作治理、回复事实校验和多模态附件消融效果。

## 生产服务骨架

第一轮生产化改造新增 FastAPI 服务和 worker 骨架，但业务读写路径仍默认使用 SQLite fallback，避免在服务边界搭建阶段同时引入数据库迁移风险。

### API

```bash
python -m pip install -e .[dev]
ticketflow-api
```

默认地址：

```text
http://127.0.0.1:8000
```

首批接口：

- `GET /healthz`
- `GET /readyz`
- `GET /api/v1/tickets`
- `GET /api/v1/tickets/{ticket_id}`
- `POST /api/v1/tickets/{ticket_id}/run`
- `GET /api/v1/tickets/{ticket_id}/audit`
- `GET /api/v1/ops/summary`

### Worker

```bash
ticketflow-worker --mode local --once
ticketflow-worker --list-tasks
```

第一轮 worker 只提供任务注册表和健康输出，预留以下任务：

- `run_ticket_workflow`
- `send_outbox_email`
- `run_claw_task`
- `build_knowledge_graph`

Celery + Redis 的异步执行会在后续阶段接入。

### Docker Compose

```bash
docker compose config
docker compose up api worker postgres redis qdrant neo4j minio
```

Compose 第一版包含：

- `api`
- `worker`
- `postgres`
- `redis`
- `qdrant`
- `neo4j`
- `minio`

这些基础设施先作为可启动服务存在；正式业务迁移会分阶段完成。

## 本地运维台

```bash
python -m pip install -e .[dev]
ticketflow-reset
ticketflow-app
```

或者直接运行：

```bash
streamlit run src/ticketflow/app.py
```

浏览器访问：

```text
http://localhost:8501
```

## 配置

复制 `.env.example` 为 `.env` 后按需修改。本地密钥、SMTP 授权码、API Key 和生成数据都不应提交到 Git。

常用生产骨架配置：

```bash
APP_ENV=development
API_HOST=127.0.0.1
API_PORT=8000
DATABASE_BACKEND=sqlite
DATABASE_URL=postgresql://ticketflow:ticketflow@localhost:5432/ticketflow
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
MINIO_ENDPOINT=http://localhost:9000
ENABLE_PRODUCTION_SERVICES=false
```

## 测试

```bash
python -m pytest -q
```

API 和 worker 骨架专项测试：

```bash
python -m pytest tests\test_api.py tests\test_worker.py -q
```

## 后续路线

1. PostgreSQL repository backend。
2. Celery + Redis 异步任务与任务状态 API。
3. Durable HITL 审批持久化与 workflow resume。
4. Skill Runtime 注册、权限、审计与版本回滚。
5. Neo4j / Graphiti 风格知识图谱记忆。
6. Production Claw Harness、轨迹记录、Pass^3 和 leaderboard。
7. OpenTelemetry、Prometheus、Grafana、Langfuse 观测体系。
8. Locust / k6 / worker 并发压测与稳定性报告。
