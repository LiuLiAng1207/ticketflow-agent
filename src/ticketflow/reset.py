from __future__ import annotations

import argparse
from pathlib import Path

from .graph import TicketFlowRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset TicketFlow SQLite data from the seeded CSVs.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TicketFlow project root.",
    )
    args = parser.parse_args()

    runner = TicketFlowRunner.from_project_root(args.project_root)
    runner.reset_demo_data()
    print(f"Reset TicketFlow demo data under {args.project_root}")


if __name__ == "__main__":
    main()
