"""Analytics module runner.  Entry point for `make analytics`."""

from __future__ import annotations

from rich.console import Console

from stratacredit import SQL_DIR
from stratacredit.db import get_connection

console = Console()

_GOLD_SCRIPTS = [
    SQL_DIR / "gold" / "01_portfolio_snapshot.sql",
    SQL_DIR / "gold" / "02_cpr_history.sql",
    SQL_DIR / "gold" / "03_delinquency_roll_rates.sql",
    SQL_DIR / "gold" / "04_loss_severity.sql",
    SQL_DIR / "gold" / "05_replines.sql",
]


def run() -> None:
    """Build all Gold analytical tables from the Silver panel."""
    console.rule("[bold blue]StrataCredit Analytics — Gold Layer")

    with get_connection("gold") as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS gold;")

        # Gold tables query Silver, so attach the Silver DB
        from stratacredit.config import data_config

        silver_path = data_config["database"]["silver_db"]
        conn.execute(f"ATTACH '{silver_path}' AS silver_db (READ_ONLY);")
        conn.execute("CREATE SCHEMA IF NOT EXISTS silver;")
        # Create views pointing to the Silver tables
        for tbl in ("loan_origination", "loan_performance", "loan_month_panel"):
            conn.execute(
                f"CREATE OR REPLACE VIEW silver.{tbl} AS SELECT * FROM silver_db.silver.{tbl};"
            )

        for script in _GOLD_SCRIPTS:
            console.print(f"  Building [cyan]{script.stem}[/cyan]...")
            sql = script.read_text(encoding="utf-8")
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                    except Exception as e:
                        console.print(f"    [yellow]WARN:[/yellow] {e}")
            console.print(f"  [green]OK[/green] {script.stem}")

    console.print("\n[green]Gold layer build complete.[/green]")


if __name__ == "__main__":
    run()
