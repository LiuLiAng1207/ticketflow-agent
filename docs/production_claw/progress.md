# Production-Claw Progress

## 2026-05-17

- Frozen interview baseline on branch `interview-stable`.
- Frozen interview baseline on tag `interview-stable-2026-05-17`.
- Created isolated development worktree for branch `production-claw-platform`.
- Baseline regression passed before production skeleton changes: `73 passed`.
- Phase 1 implementation target: FastAPI skeleton, worker skeleton, Docker Compose foundation, tests, and README update.
- Added FastAPI production skeleton endpoints, worker task registry, service settings, Dockerfile, Docker Compose, API tests, and worker tests.
- API/worker focused verification passed: `8 passed`.
- Added production control-plane persistence for workflow tasks, outbox events, idempotency keys, operation locks, approval requests, and approval decisions.
- Added Celery worker integration with eager-mode test coverage and Redis-backed Docker Compose configuration.
- Added repository protocol, PostgreSQL repository skeleton, and `ticketflow-migrate` migration runner.
- Added Streamlit sidebar visibility for recent workflow tasks, pending approvals, and outbox events.
- Verification passed after Phase 2 control-plane work: `94 passed`, `docker compose config ok`, `ticketflow-migrate --backend sqlite` ok.
- Added real production loop wiring: async workflow tasks now enter `waiting_approval` on LangGraph interrupt, approval decisions resume the workflow through API, and outbox events can be delivered idempotently by worker code.
- Updated README to describe the current API / Worker / Durable HITL / Outbox behavior in production-platform terms.
- Phase 4 PostgreSQL repository parity started: PostgreSQL now defines business tables for customers, orders, tickets, KB, policies, history, attachments, audit, workflow tasks, approvals, outbox, idempotency, and external deliveries.
- `TicketFlowRunner.from_project_root()` now uses the unified repository factory, so `DATABASE_BACKEND=postgres` selects `PostgresTicketFlowRepository` instead of hard-coded SQLite.
- Docker Compose production services now set `DATABASE_BACKEND=postgres` for API and worker; SQLite remains the local `.env.example` fallback.
- Phase 5 compose dependency scope adjusted: API/worker smoke tests depend only on PostgreSQL and Redis; Qdrant, Neo4j, and MinIO remain declared services but are not startup blockers until their own phases.
- Added repository contract tests plus optional live PostgreSQL parity test gated by `TEST_POSTGRES_DATABASE_URL`.
- Verification after Phase 4 repository parity: `100 passed`, `1 skipped` without a live PostgreSQL URL.
- Phase 5 Docker smoke verified with PostgreSQL + Redis + API + Worker: `/healthz`, `/readyz`, ticket listing, async workflow execution, durable approval resume, and Outbox delivery all ran through containerized services.
- Fixed PostgreSQL seed loading for psycopg by using cursor-level `executemany`.
- Completed Outbox queue handoff: API-created events and workflow-created high-risk side-effect events are enqueued to the Celery worker and delivered idempotently.
