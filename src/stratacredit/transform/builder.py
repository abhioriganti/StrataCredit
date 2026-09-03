"""Silver layer table builder.

Executes the three Silver SQL scripts against DuckDB to create:
  - silver.loan_origination
  - silver.loan_performance
  - silver.loan_month_panel
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from stratacredit import DATA_DIR, SQL_DIR
from stratacredit.db import get_connection

console = Console()


def bronze_has_data() -> bool:
    """Check whether the Bronze layer has any Parquet files."""
    bronze_dir = DATA_DIR / "bronze"
    return any(bronze_dir.rglob("*.parquet"))


def _exec_script(conn, script_path) -> None:
    sql = script_path.read_text(encoding="utf-8")
    # DuckDB can't always execute multiple statements in one call —
    # split on semicolons that end a statement block.
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def build_origination(conn) -> int:
    """Build silver.loan_origination and return row count."""
    _exec_script(conn, SQL_DIR / "silver" / "01_loan_origination.sql")
    result = conn.execute("SELECT COUNT(*) FROM silver.loan_origination").fetchone()
    return result[0] if result else 0


def build_performance(conn) -> int:
    """Build silver.loan_performance and return row count."""
    _exec_script(conn, SQL_DIR / "silver" / "02_loan_performance.sql")
    result = conn.execute("SELECT COUNT(*) FROM silver.loan_performance").fetchone()
    return result[0] if result else 0


def build_panel(conn) -> int:
    """Build silver.loan_month_panel and return row count."""
    _exec_script(conn, SQL_DIR / "silver" / "03_loan_month_panel.sql")
    result = conn.execute("SELECT COUNT(*) FROM silver.loan_month_panel").fetchone()
    return result[0] if result else 0


def run(force: bool = False) -> dict:
    """Build all Silver tables.

    Args:
        force: Rebuild even if tables already exist.

    Returns:
        Dict with row counts per table.
    """
    console.rule("[bold blue]StrataCredit Transform — Silver Layer")

    if not bronze_has_data():
        console.print("[red]ERROR:[/red] No Bronze Parquet files found. Run 'make ingest' first.")
        return {}

    with get_connection("silver") as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS silver;")

        console.print("Building [cyan]silver.loan_origination[/cyan]...")
        n_orig = build_origination(conn)
        console.print(f"  [green]OK[/green] {n_orig:,} rows")

        console.print("Building [cyan]silver.loan_performance[/cyan]...")
        n_perf = build_performance(conn)
        console.print(f"  [green]OK[/green] {n_perf:,} rows")

        console.print("Building [cyan]silver.loan_month_panel[/cyan]...")
        n_panel = build_panel(conn)
        console.print(f"  [green]OK[/green] {n_panel:,} rows")

        # Print summary stats
        try:
            row = conn.execute("""
                SELECT unique_loans, total_rows, earliest_vintage, latest_vintage,
                       voluntary_prepayments, credit_events, serious_delinquencies
                FROM silver.panel_summary
            """).fetchone()
            if row:
                t = Table(title="Panel Summary")
                t.add_column("Metric", style="cyan")
                t.add_column("Value", justify="right")
                t.add_row("Unique loans", f"{row[0]:,}")
                t.add_row("Total rows", f"{row[1]:,}")
                t.add_row("Vintage range", f"{row[2]}–{row[3]}")
                t.add_row("Voluntary prepayments", f"{row[4]:,}")
                t.add_row("Credit events", f"{row[5]:,}")
                t.add_row("Serious delinquencies", f"{row[6]:,}")
                console.print(t)
        except Exception:
            pass

    return {
        "loan_origination": n_orig,
        "loan_performance": n_perf,
        "loan_month_panel": n_panel,
    }


if __name__ == "__main__":
    run(force="--force" in sys.argv)
