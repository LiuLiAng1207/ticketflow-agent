from __future__ import annotations

import argparse
import json
from pathlib import Path

from .db import TicketFlowRepository
from .postgres_repository import PostgresTicketFlowRepository
from .service_settings import ServiceSettings


def run_migrations(project_root: str | Path | None = None, backend: str | None = None) -> dict[str, object]:
    settings = ServiceSettings.from_project_root(project_root)
    selected_backend = backend or settings.database_backend
    if selected_backend == "postgres":
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required for postgres migrations")
        repository = PostgresTicketFlowRepository(settings.database_url)
        repository.init_database()
        return {"status": "ok", "backend": "postgres"}

    sqlite_path = settings.project_root / "data" / "generated" / "ticketflow.sqlite3"
    repository = TicketFlowRepository(sqlite_path)
    repository.init_database()
    return {"status": "ok", "backend": "sqlite", "sqlite_path": str(sqlite_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TicketFlow production control-plane migrations.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default=None)
    args = parser.parse_args(argv)
    result = run_migrations(project_root=args.project_root, backend=args.backend)
    print(json.dumps(result, ensure_ascii=False, indent=2))
