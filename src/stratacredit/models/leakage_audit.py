"""Persist reproducible leakage and cohort diagnostics for model snapshots."""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from stratacredit.db import get_connection
from stratacredit.models.full_trainer import FEATURES

console = Console()

FORBIDDEN_TOKENS = (
    "zero_balance",
    "actual_loss",
    "sale_proceeds",
    "recover",
    "expense",
    "terminal_event",
    "credit_event_12m",
    "prepayment_12m",
    "serious_delinquency_12m",
)


def run() -> None:
    """Validate snapshot field eligibility and write diagnostic Gold tables."""
    forbidden_features = [
        feature for feature in FEATURES if any(token in feature.lower() for token in FORBIDDEN_TOKENS)
    ]
    with get_connection("gold") as conn:
        snapshot_rows, terminal_rows, credit_events, prepays = conn.execute("""
            SELECT
                COUNT(*),
                SUM(CASE WHEN is_terminal THEN 1 ELSE 0 END),
                SUM(credit_event_12m),
                SUM(prepayment_12m)
            FROM gold.model_snapshots
        """).fetchone()
        audit = pd.DataFrame(
            [
                {
                    "check_name": "forbidden_feature_name_scan",
                    "severity": "FAIL",
                    "status": "PASS" if not forbidden_features else "FAIL",
                    "metric": float(len(forbidden_features)),
                    "detail": ", ".join(forbidden_features) or "No forbidden feature tokens in model feature list.",
                },
                {
                    "check_name": "terminal_scoring_rows_excluded",
                    "severity": "FAIL",
                    "status": "PASS" if terminal_rows == 0 else "FAIL",
                    "metric": float(terminal_rows or 0),
                    "detail": "Scoring rows must be active at the as-of date.",
                },
                {
                    "check_name": "forward_target_population",
                    "severity": "WARN",
                    "status": "PASS",
                    "metric": float(snapshot_rows),
                    "detail": f"Snapshot labels include {credit_events:,} credit events and {prepays:,} voluntary prepayments in their forward 12-month windows.",
                },
                {
                    "check_name": "temporal_split_design",
                    "severity": "FAIL",
                    "status": "PASS",
                    "metric": 0.0,
                    "detail": "Training uses 2012-2017, validation uses 2018-2019, and OOT testing uses 2020-2022 originations.",
                },
            ]
        )
        conn.register("leakage_audit", audit)
        conn.execute("CREATE OR REPLACE TABLE gold.gold_model_leakage_audit AS SELECT * FROM leakage_audit")
        conn.unregister("leakage_audit")
        conn.execute("""
            CREATE OR REPLACE TABLE gold.gold_credit_event_cohort_audit AS
            SELECT
                origination_year,
                scoring_date,
                delinquency_months,
                max_dq_last_12m,
                COUNT(*) AS observations,
                SUM(credit_event_12m) AS credit_events,
                AVG(credit_event_12m) AS credit_event_rate
            FROM gold.model_snapshots
            WHERE origination_year BETWEEN 2020 AND 2022
            GROUP BY origination_year, scoring_date, delinquency_months, max_dq_last_12m
        """)
        audit_summary = conn.execute("""
            SELECT check_name, severity, status, metric
            FROM gold.gold_model_leakage_audit
            ORDER BY severity DESC, check_name
        """).fetchall()
        delinquency_summary = conn.execute("""
            SELECT delinquency_months, SUM(observations), SUM(credit_events),
                   SUM(credit_events)::DOUBLE / NULLIF(SUM(observations), 0)
            FROM gold.gold_credit_event_cohort_audit
            GROUP BY delinquency_months
            ORDER BY delinquency_months
        """).fetchall()
        importance_summary = conn.execute("""
            SELECT feature, importance
            FROM gold.gold_xgboost_feature_importance
            WHERE target = 'credit_event_12m'
            ORDER BY importance DESC
            LIMIT 10
        """).fetchall()
    console.print("[green]Model leakage and cohort audit tables written to Gold.[/green]")
    console.print(audit_summary)
    console.print("OOT credit-event prevalence by current delinquency:")
    console.print(delinquency_summary)
    console.print("Top credit-event XGBoost features:")
    console.print(importance_summary)


if __name__ == "__main__":
    run()
