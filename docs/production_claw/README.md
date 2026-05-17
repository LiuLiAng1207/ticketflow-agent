# Production-Claw Upgrade

TicketFlow is being upgraded from a local Streamlit operations prototype into a production-oriented agent platform. The first delivery phase intentionally focuses on service boundaries rather than a full data-layer migration.

## Phase 1 Scope

- Add a FastAPI backend for health checks, ticket reads, workflow triggering, audit reads, and operations summary.
- Add a worker entry point with a stable task registry.
- Add Docker Compose infrastructure for API, worker, PostgreSQL, Redis, Qdrant, Neo4j, and MinIO.
- Keep SQLite as the active fallback repository.
- Keep Streamlit as the internal operations console.

## Out Of Scope For Phase 1

- PostgreSQL business repository migration.
- Celery task execution.
- Qdrant retrieval replacement.
- Neo4j or Graphiti knowledge graph writes.
- MinIO artifact migration.
- Claw task runner and leaderboard.

These are intentionally deferred so the project can grow in controlled, reviewable increments.
