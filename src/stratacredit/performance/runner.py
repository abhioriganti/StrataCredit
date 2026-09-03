"""Performance analytics runner.  Entry point for performance analytics."""

from __future__ import annotations

from rich.console import Console

from stratacredit.db import get_connection
from stratacredit.performance.credit_events import (
    monthly_credit_event_rate,
)
from stratacredit.performance.delinquency import (
    build_transition_matrix,
    compute_roll_rates,
    delinquency_summary,
)
from stratacredit.performance.prepayment import aggregate_cpr, compute_smm
from stratacredit.performance.severity import compute_severity, severity_summary

console = Console()


def run() -> dict:
    """Run all performance analytics and return summary stats."""
    console.rule("[bold blue]StrataCredit Performance Analytics")

    console.print("Loading loan_month_panel...")
    with get_connection("silver") as conn:
        panel = conn.execute("SELECT * FROM silver.loan_month_panel").pl()

    console.print(f"  {len(panel):,} rows loaded")

    # Prepayment
    console.print("\nComputing SMM/CPR...")
    smm_df = compute_smm(panel)
    cpr_monthly = aggregate_cpr(smm_df, ["reporting_period"])
    latest_cpr = float(cpr_monthly["cpr_wa"].drop_nulls().tail(3).mean() or 0)
    console.print(f"  Latest 3-month avg CPR: {latest_cpr:.1f}%")

    # Delinquency
    console.print("\nBuilding transition matrix...")
    tm = build_transition_matrix(panel)
    compute_roll_rates(tm)
    dq_summary = delinquency_summary(panel)
    latest_dq = dq_summary.tail(1)
    if len(latest_dq) > 0:
        row = latest_dq.row(0, named=True)
        console.print(
            f"  Latest DQ30: {row.get('dq30_rate', 0):.2f}%  "
            f"DQ90+: {row.get('dq90plus_rate', 0):.2f}%"
        )

    # Credit events
    console.print("\nComputing credit event rates...")
    ce_rate = monthly_credit_event_rate(panel)
    latest_cdr = float(ce_rate["cdr"].drop_nulls().tail(3).mean() or 0)
    console.print(f"  Latest 3-month avg CDR: {latest_cdr:.2f}%")

    # Severity
    console.print("\nComputing loss severity...")
    sev = compute_severity(panel)
    sev_stats = severity_summary(sev)
    n_events = sev_stats.get("count", 0)
    wa_sev = sev_stats.get("wa_severity")
    console.print(f"  Credit events with severity: {n_events:,}")
    if wa_sev is not None:
        console.print(f"  WA loss severity: {wa_sev:.1%}")

    console.print("\n[green]Performance analytics complete.[/green]")

    return {
        "latest_cpr": latest_cpr,
        "latest_cdr": latest_cdr,
        "severity_count": n_events,
        "wa_severity": wa_sev,
    }


if __name__ == "__main__":
    run()
