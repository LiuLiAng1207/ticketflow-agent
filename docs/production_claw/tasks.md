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
- Optional live parity check: set `TEST_POSTGRES_DATABASE_URL` before running `tests/test_postgres_repository_contract.py`.
