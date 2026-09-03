"""Data quality pipeline runner.  Entry point for `make validate`."""

from __future__ import annotations

import sys
import uuid

from rich.console import Console
from rich.table import Table

from stratacredit.db import get_connection
from stratacredit.quality.reconciliation import run_all_reconciliations
from stratacredit.quality.rules import ORIGINATION_RULES, PERFORMANCE_RULES
from stratacredit.quality.validator import persist_validation_results, run_validation

console = Console()


def _print_results(results, title: str = "") -> bool:
    """Print validation results table. Returns True if any FAIL."""
    has_fail = False
    t = Table(title=title)
    t.add_column("Rule", style="cyan")
    t.add_column("Severity")
    t.add_column("Checked", justify="right")
    t.add_column("Failed", justify="right")
    t.add_column("Rate", justify="right")
    t.add_column("Status")
    for r in results:
        style = "green" if r.status == "PASS" else "yellow" if r.status == "WARN" else "red"
        t.add_row(
            r.rule_id,
            r.severity,
            f"{r.rows_checked:,}",
            f"{r.rows_failed:,}",
            f"{r.failure_rate:.2%}",
            f"[{style}]{r.status}[/{style}]",
        )
        if r.status == "FAIL":
            has_fail = True
    console.print(t)
    return has_fail


def run() -> bool:
    """Run all data quality checks. Returns True if no FAIL issues."""
    console.rule("[bold blue]StrataCredit Data Quality")
    any_fail = False
    run_id = str(uuid.uuid4())
    persisted_results = []
    persisted_quarantine = []

    # Validate origination
    console.print("\nValidating [cyan]silver.loan_origination[/cyan]...")
    try:
        with get_connection("silver") as conn:
            df = conn.execute("SELECT * FROM silver.loan_origination").pl()
        results, quarantine = run_validation(df, ORIGINATION_RULES, run_id=run_id)
        persisted_results.extend(results)
        persisted_quarantine.extend(quarantine)
        if _print_results(results, "Origination Rules"):
            any_fail = True
    except Exception as e:
        console.print(f"  [red]ERROR:[/red] {e}")
        any_fail = True

    # Validate performance
    console.print("\nValidating [cyan]silver.loan_performance[/cyan]...")
    try:
        with get_connection("silver") as conn:
            # Sample first 500k rows for speed
            df = conn.execute("SELECT * FROM silver.loan_performance LIMIT 500000").pl()
        results, quarantine = run_validation(df, PERFORMANCE_RULES, run_id=run_id)
        persisted_results.extend(results)
        persisted_quarantine.extend(quarantine)
        if _print_results(results, "Performance Rules"):
            any_fail = True
    except Exception as e:
        console.print(f"  [red]ERROR:[/red] {e}")

    # Persist before reconciliation so the workbench has an audit trail even
    # when a later check fails.
    persist_validation_results(persisted_results, persisted_quarantine)

    # Reconciliation
    console.print("\nRunning [cyan]reconciliation checks[/cyan]...")
    recon = run_all_reconciliations()
    rt = Table(title="Reconciliation")
    rt.add_column("Check")
    rt.add_column("Source", justify="right")
    rt.add_column("Target", justify="right")
    rt.add_column("Diff %", justify="right")
    rt.add_column("Status")
    for r in recon:
        style = "green" if r.status == "PASS" else "yellow" if r.status == "WARN" else "red"
        rt.add_row(
            r.check_name,
            f"{r.source_count:,}",
            f"{r.target_count:,}",
            f"{r.pct_difference:.2f}%",
            f"[{style}]{r.status}[/{style}]",
        )
        if r.note:
            rt.add_row("", "", "", "", f"  [dim]{r.note}[/dim]")
    console.print(rt)

    status = "[red]FAILED[/red]" if any_fail else "[green]PASSED[/green]"
    console.print(f"\nData quality: {status}")
    return not any_fail


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
