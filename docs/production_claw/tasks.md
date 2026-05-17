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
