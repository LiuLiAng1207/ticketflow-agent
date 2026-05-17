from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import TicketFlowSettings
from .db import TicketFlowRepository
from .graph import TicketFlowRunner
from .models import TicketRecord
from .seed import build_seed_dataset


def _category_to_action(category: str | None) -> str:
    if category == "billing_refund":
        return "refund"
    if category in {"technical_issue", "delivery_issue"}:
        return "escalation"
    if category == "account_access":
        return "troubleshoot"
    return "status_update"


def _target_rows(repository: TicketFlowRepository, tickets: list[TicketRecord], limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ticket in tickets:
        query = f"{ticket.title}\n{ticket.body}\n{ticket.product}"
        action_type = _category_to_action(ticket.expected_category)
        policies = repository.lookup_policy(action_type, query=query, limit=1)
        if policies:
            rows.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "source_type": "policy",
                    "target_doc_id": f"policy:{policies[0]['policy_id']}",
                    "query_type": "policy_for_action",
                }
            )
        history = repository.get_ticket_history(ticket.ticket_id, limit=1)
        if history:
            rows.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "source_type": "history",
                    "target_doc_id": f"history:{history[0]['event_id']}",
                    "query_type": "same_ticket_history",
                }
            )
        kb_hits = repository.search_kb(query, ticket.product, limit=1)
        if kb_hits:
            rows.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "source_type": "kb",
                    "target_doc_id": f"kb:{kb_hits[0]['doc_id']}",
                    "query_type": "product_kb",
                }
            )
        if len(rows) >= limit:
            break
    return rows[:limit]


def _calculate_ranking_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"hit_rate": 0.0, "mrr": 0.0, "error_rate": 1.0}
    hits = [item for item in results if item["hit"]]
    reciprocal_ranks = [1.0 / item["rank"] for item in hits if item.get("rank")]
    hit_rate = len(hits) / len(results)
    return {
        "hit_rate": round(hit_rate, 3),
        "mrr": round(sum(reciprocal_ranks) / len(results), 3),
        "error_rate": round(1.0 - hit_rate, 3),
    }


def run_retrieval_strategy_eval(
    *,
    project_root: Path,
    sample_size: int = 600,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    settings = TicketFlowSettings.from_project_root(project_root)
    if not (settings.seed_dir / "tickets.csv").exists():
        build_seed_dataset(settings.seed_dir)
    repository = TicketFlowRepository(settings.db_path)
    repository.bootstrap(settings.seed_dir)

    runner = TicketFlowRunner.from_project_root(project_root, overrides=overrides)
    try:
        tickets = runner.list_open_tickets(limit=1000)
        targets = _target_rows(repository, tickets, sample_size)
        sample_results: list[dict[str, Any]] = []
        for row in targets:
            ticket = runner.get_ticket(row["ticket_id"])
            retrieved = runner.retriever.retrieve(ticket)
            doc_ids = retrieved.stats.retrieved_doc_ids
            target_doc_id = row["target_doc_id"]
            rank = doc_ids.index(target_doc_id) + 1 if target_doc_id in doc_ids else None
            sample_results.append(
                {
                    **row,
                    "hit": rank is not None,
                    "rank": rank,
                    "retrieved_doc_ids": doc_ids,
                    "retrieval_metadata": retrieved.stats.metadata,
                }
            )
    finally:
        runner.close()

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sample_results:
        by_source[item["source_type"]].append(item)

    return {
        "project": "TicketFlow retrieval strategy evaluation",
        "sample_size": len(sample_results),
        "target_sample_size": sample_size,
        "metrics": _calculate_ranking_metrics(sample_results),
        "by_source": {source: _calculate_ranking_metrics(items) for source, items in sorted(by_source.items())},
        "retrieval_stack": sample_results[0]["retrieval_metadata"] if sample_results else {},
        "samples": sample_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TicketFlow retrieval HitRate/MRR/error rate.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "retrieval_strategy_eval_report_v1.json",
    )
    args = parser.parse_args()
    report = run_retrieval_strategy_eval(project_root=args.project_root, sample_size=args.sample_size)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("project", "sample_size", "metrics", "by_source", "retrieval_stack")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
