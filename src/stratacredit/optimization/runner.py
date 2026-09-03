"""Pool selection runner.  Entry point for `make pool`."""

from __future__ import annotations

from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from stratacredit.config import pool_config
from stratacredit.db import get_connection
from stratacredit.models.asof_scoring import score_as_of
from stratacredit.optimization.constraints import check_pool_constraints
from stratacredit.optimization.eligibility import apply_eligibility_filters
from stratacredit.optimization.selector import (
    compare_candidate_vs_selected,
    score_loans,
    select_pool_greedy,
)

console = Console()


def run(cutoff_date: date | None = None) -> dict:
    """Run pool selection for a given cutoff date.

    Args:
        cutoff_date: As-of date. Defaults to latest period in panel.

    Returns:
        Dict with candidate, selected DataFrames and constraint results.
    """
    console.rule("[bold blue]StrataCredit Pool Selection")

    with get_connection("silver") as conn:
        if cutoff_date is None:
            row = conn.execute(
                "SELECT MAX(reporting_period) FROM silver.loan_month_panel WHERE is_terminal = FALSE"
            ).fetchone()
            cutoff_date = row[0] if row else date.today()

        console.print(f"Cutoff date: {cutoff_date}")
        panel_snapshot = conn.execute(f"""
            SELECT * FROM silver.loan_month_panel
            WHERE reporting_period = CAST('{cutoff_date}' AS DATE)
              AND is_terminal = FALSE
        """).pl()

    console.print(f"Universe: {len(panel_snapshot):,} loans")

    # Step 1: Eligibility filters
    candidates, exclusions = apply_eligibility_filters(panel_snapshot)
    n_cand = exclusions["_candidate_count"]
    n_excl = exclusions["_total_excluded"]
    console.print(f"After eligibility: {n_cand:,} candidates ({n_excl:,} excluded)")
    for reason, cnt in exclusions.items():
        if not reason.startswith("_") and cnt > 0:
            console.print(f"  {reason}: {cnt:,}")

    if n_cand == 0:
        console.print("[red]No eligible loans found.[/red]")
        return {}

    # Step 2: Attach model probabilities computed strictly as of the cutoff.
    risk_scores = score_as_of(cutoff_date)
    candidates = candidates.join(risk_scores, on="loan_id", how="left")
    candidates = candidates.with_columns(
        [
            pl.col("credit_event_score").fill_null(0.0),
            pl.col("prepayment_score").fill_null(0.0),
        ]
    )
    candidates = candidates.with_columns(
        (
            pl.col("current_upb")
            * pl.col("credit_event_score")
            * pl.col("loss_severity_score").fill_null(0.35)
        ).alias("expected_loss_usd")
    )
    candidates = score_loans(
        candidates, "credit_event_score", "prepayment_score", "expected_loss_usd"
    )

    # Step 3: Select pool
    target_upb = pool_config["pool"]["target_upb"]
    tol_pct = pool_config["pool"]["upb_tolerance_pct"]
    console.print(f"\nSelecting pool: target UPB = ${target_upb:,.0f}")
    selected = select_pool_greedy(candidates, target_upb, tol_pct)
    sel_upb = float(selected["current_upb"].sum())
    console.print(f"Selected: {len(selected):,} loans  UPB = ${sel_upb:,.0f}")

    # Step 4: Constraint reconciliation
    console.print("\nChecking constraints...")
    constraint_results = check_pool_constraints(selected)
    passes = sum(1 for r in constraint_results if r.status == "PASS")
    fails = sum(1 for r in constraint_results if r.status == "FAIL")

    ct = Table(title="Constraint Reconciliation")
    ct.add_column("Constraint")
    ct.add_column("Target", justify="right")
    ct.add_column("Actual", justify="right")
    ct.add_column("Status")
    for r in constraint_results:
        style = "green" if r.status == "PASS" else "red"
        ct.add_row(
            r.name, str(r.target or "—"), f"{r.actual:.2f}", f"[{style}]{r.status}[/{style}]"
        )
    console.print(ct)
    console.print(f"\n{passes} PASS / {fails} FAIL")

    # Step 5: Comparison table
    comparison = compare_candidate_vs_selected(candidates, selected)
    console.print("\n[bold]Candidate vs Selected:[/bold]")
    # Polars renders Unicode table glyphs that fail in legacy Windows consoles.
    console.print(comparison.to_pandas().to_string(index=False))

    return {
        "candidates": candidates,
        "selected": selected,
        "constraint_results": constraint_results,
        "comparison": comparison,
        "cutoff_date": cutoff_date,
    }


if __name__ == "__main__":
    run()
