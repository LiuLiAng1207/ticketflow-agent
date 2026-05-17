# TicketFlow Agent Development Notes

## Version Safety

- `main` is the interview-stable baseline unless explicitly changed.
- `interview-stable` and `interview-stable-2026-05-17` preserve the current demo baseline.
- Production platform work must happen on `production-claw-platform` or a branch/worktree forked from it.
- Do not commit `.env`, generated databases, local logs, model weights, caches, or API keys.

## Current Phase

Phase 1 only adds the production service skeleton:

- FastAPI API gateway.
- Local worker skeleton.
- Docker Compose infrastructure placeholders.
- SQLite fallback remains the active business repository.

Do not migrate business reads/writes to PostgreSQL, Qdrant, Neo4j, MinIO, or Celery in this phase unless a later plan explicitly says so.

## Verification

Before claiming completion, run:

```powershell
python -m pytest -q
```

For API-only changes, also run:

```powershell
python -m pytest tests\test_api.py tests\test_worker.py -q
```

## Collaboration Rules

- Keep changes small and commit by phase.
- Prefer adding stable interfaces over replacing existing Streamlit behavior.
- If a file contains unrelated user changes, do not revert them.
- If a change affects external side effects such as email, refunds, or approvals, add an idempotency note or test before implementation.
