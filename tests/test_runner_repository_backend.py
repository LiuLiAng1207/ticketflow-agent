from __future__ import annotations

from ticketflow.db import TicketFlowRepository
from ticketflow.graph import TicketFlowRunner


def test_runner_factory_uses_service_repository_backend(monkeypatch, ticketflow_project):
    from ticketflow import graph

    calls: list[dict[str, object]] = []

    def fake_create_repository(settings, *, sqlite_db_path=None):
        calls.append(
            {
                "database_backend": settings.database_backend,
                "database_url": settings.database_url,
                "sqlite_db_path": sqlite_db_path,
            }
        )
        return TicketFlowRepository(sqlite_db_path)

    monkeypatch.setattr(graph, "create_repository", fake_create_repository, raising=False)

    runner = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={
            "DATABASE_BACKEND": "postgres",
            "DATABASE_URL": "postgresql://ticketflow:ticketflow@localhost:5432/ticketflow",
            "RAG_EMBED_BACKEND": "hash",
        },
    )
    try:
        assert calls
        assert calls[0]["database_backend"] == "postgres"
        assert calls[0]["database_url"] == "postgresql://ticketflow:ticketflow@localhost:5432/ticketflow"
    finally:
        runner.close()
