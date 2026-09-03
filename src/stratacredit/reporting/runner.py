"""Reporting runner.  Entry point for `make report`."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from stratacredit.analytics.metrics import compute_portfolio_metrics
from stratacredit.analytics.replines import build_replines
from stratacredit.optimization.runner import run as run_pool
from stratacredit.reporting.exports import (
    export_portfolio_summary,
    export_replines,
    export_selected_pool,
)
from stratacredit.reporting.memo import generate_memo
from stratacredit.scenarios.engine import run_scenarios

console = Console()


def run() -> None:
    """Generate all reports and exports."""
    console.rule("[bold blue]StrataCredit Reporting")

    # Pool selection
    pool_result = run_pool()
    if not pool_result:
        console.print("[red]Pool selection failed.[/red]")
        return

    selected = pool_result["selected"]
    candidates = pool_result["candidates"]
    constraints = pool_result["constraint_results"]
    cutoff = pool_result["cutoff_date"]

    # Metrics
    sel_metrics = compute_portfolio_metrics(selected)
    cand_metrics = compute_portfolio_metrics(candidates)

    # Scenarios
    scenarios = run_scenarios(
        pool=selected,
        base_credit_event_rate=0.015,
        base_severity=0.35,
        base_cpr=0.18,
        ce_assumption_pct=4.0,
    )

    # Replines
    replines = build_replines(selected)

    # Exports
    Path("outputs").mkdir(exist_ok=True)

    p1 = export_selected_pool(selected)
    console.print(f"  [green]✓[/green] {p1}")

    p2 = export_replines(replines)
    console.print(f"  [green]✓[/green] {p2}")

    p3 = export_portfolio_summary(sel_metrics)
    console.print(f"  [green]✓[/green] {p3}")

    p4 = generate_memo(
        metrics=sel_metrics,
        constraint_results=constraints,
        scenario_results=scenarios,
        as_of_date=cutoff,
        candidate_metrics=cand_metrics,
    )
    console.print(f"  [green]✓[/green] {p4}")

    console.print("\n[green]All reports generated in outputs/[/green]")


if __name__ == "__main__":
    run()
