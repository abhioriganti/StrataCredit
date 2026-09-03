"""Disk-backed, leakage-safe model snapshot construction."""

from __future__ import annotations

from rich.console import Console

from stratacredit.config import data_config
from stratacredit.db import get_connection

console = Console()


def build_model_snapshots(rebuild: bool = False) -> None:
    """Materialize quarterly snapshots in DuckDB without loading the panel in Python."""
    silver_path = data_config["database"]["silver_db"]
    with get_connection("gold") as conn:
        exists = conn.execute(
            """SELECT COUNT(*) FROM information_schema.tables
               WHERE table_schema = 'gold' AND table_name = 'model_snapshots'"""
        ).fetchone()[0]
        if exists and not rebuild:
            count = conn.execute("SELECT COUNT(*) FROM gold.model_snapshots").fetchone()[0]
            console.print(f"[green]Reusing persisted model snapshots: {count:,} rows.[/green]")
            return
        conn.execute(f"ATTACH '{silver_path}' AS silver_model (READ_ONLY)")
        console.print("Building disk-backed quarterly model snapshots...")
        conn.execute("""
            CREATE OR REPLACE TABLE gold.model_snapshots AS
            WITH panel_features AS (
                SELECT
                    loan_id, reporting_period AS scoring_date, origination_year,
                    is_terminal,
                    current_upb, original_upb, original_interest_rate,
                    current_interest_rate, fico, original_ltv, original_cltv, dti,
                    loan_age, remaining_term, delinquency_months, estimated_ltv,
                    CAST(modification_flag AS INTEGER) AS modification_flag,
                    CASE loan_purpose WHEN 'P' THEN 0 WHEN 'C' THEN 1 WHEN 'R' THEN 2 ELSE 3 END AS loan_purpose_code,
                    CASE occupancy WHEN 'P' THEN 0 WHEN 'S' THEN 1 WHEN 'I' THEN 2 ELSE 3 END AS occupancy_code,
                    CASE property_type WHEN 'SF' THEN 0 WHEN 'CO' THEN 1 WHEN 'MH' THEN 2 WHEN 'PU' THEN 3 ELSE 4 END AS property_type_code,
                    EXTRACT(MONTH FROM reporting_period) AS reporting_month,
                    SUM(CASE WHEN delinquency_months >= 1 THEN 1 ELSE 0 END)
                        OVER loan_window AS dq30_months_last_12m,
                    SUM(CASE WHEN delinquency_months >= 2 THEN 1 ELSE 0 END)
                        OVER loan_window AS dq60_months_last_12m,
                    SUM(CASE WHEN delinquency_months >= 3 THEN 1 ELSE 0 END)
                        OVER loan_window AS dq90_months_last_12m,
                    MAX(delinquency_months) OVER loan_window AS max_dq_last_12m,
                    MAX(CAST(modification_flag AS INTEGER)) OVER loan_window AS ever_modified_last_12m,
                    MAX(CASE WHEN is_credit_event THEN 1 ELSE 0 END)
                        OVER forward_window AS credit_event_12m,
                    MAX(CASE WHEN is_voluntary_prepayment THEN 1 ELSE 0 END)
                        OVER forward_window AS prepayment_12m,
                    MAX(CASE WHEN is_seriously_delinquent THEN 1 ELSE 0 END)
                        OVER forward_window AS serious_delinquency_12m,
                    COUNT(*) OVER forward_window AS forward_months
                FROM silver_model.silver.loan_month_panel
                WINDOW
                    loan_window AS (PARTITION BY loan_id ORDER BY reporting_period ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
                    forward_window AS (PARTITION BY loan_id ORDER BY reporting_period ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING)
            )
            SELECT * EXCLUDE (forward_months)
            FROM panel_features
            WHERE reporting_month IN (3, 6, 9, 12)
              -- A loan which terminates within the horizon has fewer than
              -- twelve subsequent panel rows.  Keep those positive examples;
              -- requiring a full horizon for them would erase payoff and
              -- credit-event labels from the training population.
              AND (
                  forward_months >= 12
                  OR credit_event_12m = 1
                  OR prepayment_12m = 1
                  OR serious_delinquency_12m = 1
              )
              AND NOT is_terminal
        """)
        count = conn.execute("SELECT COUNT(*) FROM gold.model_snapshots").fetchone()[0]
        console.print(f"[green]Model snapshots ready: {count:,} rows.[/green]")


if __name__ == "__main__":
    build_model_snapshots(rebuild=True)
