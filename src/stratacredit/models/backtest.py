"""Display the persisted out-of-time model backtests."""

from __future__ import annotations

from rich.console import Console

from stratacredit.db import get_connection


def run() -> None:
    console = Console()
    with get_connection("gold") as conn:
        for table in ("gold_credit_event_12m_backtest", "gold_prepayment_12m_backtest"):
            try:
                console.print(f"[bold]{table}[/bold]")
                console.print(
                    conn.execute(f"SELECT * FROM gold.{table} ORDER BY scoring_date LIMIT 20").df()
                )
            except Exception:
                console.print(f"[yellow]{table} is not available; run `make train` first.[/yellow]")


if __name__ == "__main__":
    run()
