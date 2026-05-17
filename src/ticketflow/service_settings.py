from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import _load_local_env


def _get_bool(name: str, default: bool, overrides: dict[str, str] | None = None) -> bool:
    raw = _get_env(name, overrides)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_env(name: str, overrides: dict[str, str] | None = None, default: str | None = None) -> str | None:
    if overrides and name in overrides:
        return overrides[name]
    return os.getenv(name, default)


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
    celery_broker_url: str | None
    celery_result_backend: str | None
    celery_task_always_eager: bool
    skill_registry_backend: str = "repository"
    skills_dir: Path | None = None
    enable_skill_runtime: bool = True
    claw_tasks_dir: Path | None = None
    neo4j_user: str | None = "neo4j"
    neo4j_password: str | None = None
    kg_backend: str = "disabled"

    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path | None = None,
        overrides: dict[str, str] | None = None,
    ) -> "ServiceSettings":
        root = Path(project_root) if project_root is not None else Path.cwd()
        _load_local_env(root)
        app_env = _get_env("APP_ENV", overrides, "development") or "development"
        raw_skills_dir = _get_env("SKILLS_DIR", overrides, "skills") or "skills"
        skills_dir = Path(raw_skills_dir)
        if not skills_dir.is_absolute():
            skills_dir = root / skills_dir
            source_tree_skills_dir = Path(__file__).resolve().parents[2] / raw_skills_dir
            if not skills_dir.exists() and source_tree_skills_dir.exists():
                skills_dir = source_tree_skills_dir
        raw_claw_tasks_dir = _get_env("CLAW_TASKS_DIR", overrides, "claw_tasks") or "claw_tasks"
        claw_tasks_dir = Path(raw_claw_tasks_dir)
        if not claw_tasks_dir.is_absolute():
            claw_tasks_dir = root / claw_tasks_dir
            source_tree_claw_tasks_dir = Path(__file__).resolve().parents[2] / raw_claw_tasks_dir
            if not claw_tasks_dir.exists() and source_tree_claw_tasks_dir.exists():
                claw_tasks_dir = source_tree_claw_tasks_dir
        return cls(
            project_root=root,
            app_env=app_env,
            api_host=_get_env("API_HOST", overrides, "127.0.0.1") or "127.0.0.1",
            api_port=int(_get_env("API_PORT", overrides, "8000") or "8000"),
            database_backend=(_get_env("DATABASE_BACKEND", overrides, "sqlite") or "sqlite").strip().lower(),
            database_url=_get_env("DATABASE_URL", overrides),
            redis_url=_get_env("REDIS_URL", overrides),
            qdrant_url=_get_env("QDRANT_URL", overrides),
            neo4j_uri=_get_env("NEO4J_URI", overrides),
            neo4j_user=_get_env("NEO4J_USER", overrides, "neo4j"),
            neo4j_password=_get_env("NEO4J_PASSWORD", overrides),
            kg_backend=(_get_env("KG_BACKEND", overrides, "disabled") or "disabled").strip().lower(),
            minio_endpoint=_get_env("MINIO_ENDPOINT", overrides),
            enable_production_services=_get_bool("ENABLE_PRODUCTION_SERVICES", False, overrides),
            celery_broker_url=_get_env("CELERY_BROKER_URL", overrides) or _get_env("REDIS_URL", overrides),
            celery_result_backend=_get_env("CELERY_RESULT_BACKEND", overrides) or _get_env("REDIS_URL", overrides),
            celery_task_always_eager=_get_bool("CELERY_TASK_ALWAYS_EAGER", app_env != "production", overrides),
            skill_registry_backend=(
                _get_env("SKILL_REGISTRY_BACKEND", overrides, "repository") or "repository"
            ).strip().lower(),
            skills_dir=skills_dir.resolve(),
            enable_skill_runtime=_get_bool("ENABLE_SKILL_RUNTIME", True, overrides),
            claw_tasks_dir=claw_tasks_dir.resolve(),
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
                "knowledge_graph": {
                    "backend": self.kg_backend,
                    "configured": self.kg_backend != "disabled",
                    "neo4j_uri": self.neo4j_uri,
                },
                "minio": {"configured": bool(self.minio_endpoint), "endpoint": self.minio_endpoint},
                "celery": {
                    "configured": bool(self.celery_broker_url),
                    "broker_url": self.celery_broker_url,
                    "task_always_eager": self.celery_task_always_eager,
                },
                "skill_runtime": {
                    "enabled": self.enable_skill_runtime,
                    "registry_backend": self.skill_registry_backend,
                    "skills_dir": str(self.skills_dir) if self.skills_dir else None,
                },
                "claw_harness": {
                    "tasks_dir": str(self.claw_tasks_dir) if self.claw_tasks_dir else None,
                },
            },
        }
