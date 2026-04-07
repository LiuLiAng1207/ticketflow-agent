from __future__ import annotations

from pathlib import Path

import pytest

from ticketflow.graph import TicketFlowRunner
from ticketflow.seed import build_seed_dataset


@pytest.fixture()
def ticketflow_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "ticketflow"
    (project_root / "data" / "seed").mkdir(parents=True)
    build_seed_dataset(project_root / "data" / "seed")
    return project_root


@pytest.fixture()
def runner(ticketflow_project: Path) -> TicketFlowRunner:
    return TicketFlowRunner.from_project_root(ticketflow_project)


@pytest.fixture()
def runner_factory(ticketflow_project: Path):
    def _make(**overrides: str) -> TicketFlowRunner:
        return TicketFlowRunner.from_project_root(ticketflow_project, overrides=overrides or None)

    return _make
