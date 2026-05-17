# Phase 1 Task Board

## API Skeleton

- `GET /healthz`
- `GET /readyz`
- `GET /api/v1/tickets`
- `GET /api/v1/tickets/{ticket_id}`
- `POST /api/v1/tickets/{ticket_id}/run`
- `GET /api/v1/tickets/{ticket_id}/audit`
- `GET /api/v1/ops/summary`

## Worker Skeleton

- Local worker CLI.
- Task registry.
- JSON-friendly health output.
- Reserved task names for workflow, outbox email, Claw, and knowledge graph.

## Infrastructure

- Dockerfile.
- Docker Compose for API, worker, PostgreSQL, Redis, Qdrant, Neo4j, and MinIO.
- `.env.example` production service settings.

## Verification

- API smoke tests.
- Worker registry tests.
- Full pytest regression.

## Phase 2 Control Plane

- Workflow task lifecycle is persisted in `workflow_tasks`.
- External side effects are staged through `outbox_events`.
- Duplicate operations are guarded by `idempotency_keys` and `external_operation_locks`.
- Human approval state is persisted in `approval_requests` and `approval_decisions`.
- Celery eager mode is used in local tests; Docker Compose config runs a Redis-backed Celery worker.

## Phase 3 Production Loop

- Async workflow execution records LangGraph approval interrupts as `waiting_approval` tasks.
- Approval API decisions resume the stored workflow and complete or fail the associated task.
- Outbox worker delivery writes external delivery records once and skips duplicate delivery attempts.
- Streamlit production sidebar includes queued, running, and `waiting_approval` tasks in the pending count.

## Phase 4 PostgreSQL Repository Parity

- PostgreSQL schema now covers the business read path: customers, orders, tickets, KB articles, policies, reply templates, history, attachments, and attachment evidence.
- PostgreSQL schema now covers the control plane: audit log, workflow tasks, approvals, outbox, idempotency keys, operation locks, and external email deliveries.
- PostgreSQL repository exposes the same core workflow methods as SQLite for RAG reads, action tools, task state, durable approvals, outbox, and email delivery records.
- Runner construction uses the repository factory, so production can switch with `DATABASE_BACKEND=postgres` and `DATABASE_URL=...`.
- Docker Compose API and worker services now select PostgreSQL by default; `.env.example` keeps SQLite as a safe local fallback.
- API and worker startup depend only on PostgreSQL and Redis for the production-loop smoke test; Qdrant, Neo4j, and MinIO stay available as opt-in services for later phases.
- Optional live parity check: set `TEST_POSTGRES_DATABASE_URL` before running `tests/test_postgres_repository_contract.py`.

## Phase 5 Docker Production Loop

- `docker compose up -d postgres redis api worker` starts the production loop without requiring optional Qdrant, Neo4j, or MinIO services.
- API readiness bootstraps the PostgreSQL seed data and reports Redis/Celery configuration.
- `POST /api/v1/tickets/{ticket_id}/run` creates a Redis-backed Celery task and persists status transitions in PostgreSQL.
- High-risk refund workflows enter `waiting_approval`, persist approval payload snapshots, and resume through `POST /api/v1/approvals/{approval_id}/decision`.
- Workflow-created Outbox events are enqueued for delivery after approval resume; API-created Outbox events are enqueued immediately.
- Outbox delivery uses the worker plus operation locks to avoid duplicate external side effects.

## Phase 6 API-First Ops Console

- Streamlit production control panel now targets the FastAPI service first instead of reading local repository state by default.
- The panel displays API availability, API URL, database backend, Celery mode, task queue, approval queue, and Outbox queue.
- If FastAPI is offline, the panel falls back to the local repository and shows the fallback reason instead of breaking the UI.
- `TICKETFLOW_API_URL` or `API_BASE_URL` can override the API address; otherwise `API_HOST` and `API_PORT` are used.

## Phase 7 Conversational Ticket Agent

- Added `POST /api/v1/agent/chat` as a safe conversational operations entrypoint.
- Supported intents: query ticket, create ticket, run ticket workflow, list pending approvals, list Outbox events, and submit knowledge candidates.
- High-risk approvals are intentionally not executable directly through chat; the Agent redirects operators to durable approval surfaces.
- Newly created chat tickets are persisted in the same repository backend and get audit events tagged as `ticket_created_from_chat`.

## Phase 8 Knowledge Graph / GraphRAG Foundation

- Added a `KnowledgeGraphStore` abstraction with disabled, in-memory, and Neo4j backends.
- Added idempotent ticket graph construction from tickets, customers, orders, products, retrieved evidence, workflow tasks, approvals, and Outbox/email events.
- Added KG API endpoints: `/api/v1/kg/health`, `/api/v1/kg/tickets/{ticket_id}/rebuild`, `/api/v1/kg/tickets/{ticket_id}`, and `/api/v1/kg/search`.
- Added worker support for `build_knowledge_graph` so KG rebuild can move into async production queues.
- Added conversational Agent intent for explaining a ticket decision and evidence chain through the graph.
- Added Streamlit ops visibility for KG health and a per-ticket knowledge graph explanation panel.
- KG is an explanation and retrieval aid only; it does not bypass sufficiency checks, tool approval, or durable HITL.

## Phase 9 Skill Runtime

- Added a Skill package contract based on `skill.yaml` and `SKILL.md`.
- Added repository-backed Skill Registry tables and methods for SQLite and PostgreSQL.
- Added audited Skill Runtime execution with enable/disable checks, allowlisted executors, idempotency hook, and `skill_runs`.
- Added built-in skills: `ticketflow-ops`, `ticketflow-kg-memory`, `ticketflow-safety-governance`, `ticketflow-batch-ops`, and `ticketflow-claw-eval`.
- Added Skill API endpoints for reload, list, detail, enable, disable, run, and run history.
- Added worker support for `run_skill` and conversational Agent routing through Skill Runtime for evidence-chain explanation and batch low-risk operations.
- Streamlit production control panel now surfaces Skill Runtime status and recent runs.

## Phase 10 Production Claw Harness

- Added a Claw task contract based on `claw_tasks/*.yaml`.
- Added repository-backed Claw task, run, attempt, trajectory, and score tables for SQLite and PostgreSQL.
- Added deterministic Claw Runtime scoring for completion, safety, tool correctness, argument correctness, RAG grounding, approval correctness, and trajectory quality.
- Added built-in tasks for ticket evidence query, chat-created tickets, governed refund workflow, approval inspection, and KG explanation.
- Added Claw API endpoints for reload, task list, task detail, sync/async run, run detail, and leaderboard.
- Added worker support for `run_claw_task` and Skill Runtime integration through `ticketflow-claw-eval`.
- Streamlit production control panel now surfaces Claw task and leaderboard snapshots.
