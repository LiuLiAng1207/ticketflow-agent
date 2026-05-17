from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - optional dependency fallback
    SqliteSaver = None  # type: ignore[assignment]


@dataclass(slots=True)
class CheckpointerHandle:
    saver: Any
    backend: str
    close: Callable[[], None] | None = None


def build_checkpointer(
    *,
    backend: str,
    sqlite_path: Path,
    serializer: JsonPlusSerializer,
) -> CheckpointerHandle:
    normalized_backend = (backend or "sqlite").strip().lower()

    if normalized_backend == "sqlite" and SqliteSaver is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return CheckpointerHandle(
            saver=SqliteSaver(connection, serde=serializer),
            backend="sqlite",
            close=connection.close,
        )

    return CheckpointerHandle(
        saver=InMemorySaver(serde=serializer),
        backend="memory",
        close=None,
    )
