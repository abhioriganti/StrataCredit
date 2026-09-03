"""Scenarios runner.  Entry point for `make scenarios`."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from stratacredit.optimization.runner import run as run_pool
from stratacredit.scenarios.engine import run_scenarios, scenarios_to_dataframe

console = Console()


def run() -> list:
    """Run stress scenarios against the selected pool."""
    console.rule("[bold blue]StrataCredit Stress Scenarios")

    # Run pool selection first (or load pre-selected pool)
    pool_result = run_pool()
    if not pool_result or "selected" not in pool_result:
        console.print("[red]No selected pool found. Run pool selection first.[/red]")
        return []

    selected = pool_result["selected"]
    pool_upb = float(selected["current_upb"].sum())

    # Placeholder base rates — in production these come from model scores
    # attached to the selected pool during optimization
    base_ce_rate = float(selected["credit_event_score"].mean() or 0.0)
    base_severity = float(selected["loss_severity_score"].mean() or 0.35)
    base_cpr = float(selected["prepayment_score"].mean() or 0.0)

    console.print(f"\nPool UPB: ${pool_upb:,.0f}")
    console.print(f"Base credit event rate: {base_ce_rate:.1%}")
    console.print(f"Base severity: {base_severity:.1%}")
    console.print(f"Base CPR: {base_cpr:.1%}")

    results = run_scenarios(
        pool=selected,
        base_credit_event_rate=base_ce_rate,
        base_severity=base_severity,
        base_cpr=base_cpr,
        ce_assumption_pct=4.0,  # example 4% CE assumption
    )

    df = scenarios_to_dataframe(results)

    t = Table(title="Stress Scenario Results")
    for col in df.columns:
        t.add_column(col, justify="right" if col != "Scenario" else "left")
    for row in df.iter_rows():
        t.add_row(*[str(v) if v is not None else "—" for v in row])
    console.print(t)

    console.print(
        "\n[dim]Note: These are deterministic sensitivity scenarios, not rating-agency models.[/dim]"
    )
    return results


if __name__ == "__main__":
    run()
