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
