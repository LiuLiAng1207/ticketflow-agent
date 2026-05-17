from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import _load_local_env


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class ServiceSettings:
    project_root: Path
    app_env: str
    api_host: str
    api_port: int
    database_backend: str
    database_url: str | None
    redis_url: str | None
    qdrant_url: str | None
    neo4j_uri: str | None
    minio_endpoint: str | None
    enable_production_services: bool

    @classmethod
    def from_project_root(cls, project_root: str | Path | None = None) -> "ServiceSettings":
        root = Path(project_root) if project_root is not None else Path.cwd()
        _load_local_env(root)
        return cls(
            project_root=root,
            app_env=os.getenv("APP_ENV", "development"),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            database_backend=os.getenv("DATABASE_BACKEND", "sqlite").strip().lower(),
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            qdrant_url=os.getenv("QDRANT_URL"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            minio_endpoint=os.getenv("MINIO_ENDPOINT"),
            enable_production_services=_get_bool("ENABLE_PRODUCTION_SERVICES", False),
        )

    def readiness_payload(self, *, sqlite_db_path: Path | None = None) -> dict[str, object]:
        database: dict[str, object] = {
            "backend": self.database_backend,
            "configured": bool(self.database_url) if self.database_backend != "sqlite" else True,
        }
        if sqlite_db_path is not None:
            database["sqlite_path"] = str(sqlite_db_path)
            database["sqlite_exists"] = sqlite_db_path.exists()

        return {
            "status": "ready",
            "app_env": self.app_env,
            "production_services_enabled": self.enable_production_services,
            "database": database,
            "dependencies": {
                "redis": {"configured": bool(self.redis_url), "url": self.redis_url},
                "qdrant": {"configured": bool(self.qdrant_url), "url": self.qdrant_url},
                "neo4j": {"configured": bool(self.neo4j_uri), "uri": self.neo4j_uri},
                "minio": {"configured": bool(self.minio_endpoint), "endpoint": self.minio_endpoint},
            },
        }
