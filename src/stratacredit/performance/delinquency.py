"""Delinquency analytics: transition matrices and roll rates.

Delinquency States
------------------
CURRENT   - 0 months past due
DQ30      - 30 days past due (1 month)
DQ60      - 60 days past due (2 months)
DQ90PLUS  - 90+ days past due (3+ months)
REO       - Real estate owned
TERMINAL  - Zero balance / terminated

Transition Matrix
-----------------
For each loan in month t, records (state_t → state_{t+1}).
Transition rates are expressed as % of loans in state_t.

Key transitions tracked:
  CURRENT  → DQ30       (new delinquency rate)
  DQ30     → CURRENT    (cure rate from 30)
  DQ30     → DQ60       (roll-forward from 30)
  DQ60     → DQ30       (cure rate from 60)
  DQ60     → DQ90PLUS   (roll-forward from 60)
  DQ90PLUS → CURRENT    (cure from serious DQ)
  DQ90PLUS → TERMINAL   (credit event from serious DQ)
"""

from __future__ import annotations

import polars as pl

STATES = ["CURRENT", "DQ30", "DQ60", "DQ90PLUS", "REO", "TERMINAL"]

KEY_TRANSITIONS = [
    ("CURRENT", "DQ30", "new_delinquency_rate"),
    ("DQ30", "CURRENT", "cure_rate_30"),
    ("DQ30", "DQ60", "roll_forward_30_to_60"),
    ("DQ60", "DQ30", "cure_rate_60"),
    ("DQ60", "DQ90PLUS", "roll_forward_60_to_90plus"),
    ("DQ90PLUS", "CURRENT", "cure_rate_90plus"),
    ("DQ90PLUS", "TERMINAL", "credit_event_from_90plus"),
]


def build_transition_matrix(panel: pl.DataFrame) -> pl.DataFrame:
    """Build monthly delinquency transition matrix.

    Args:
        panel: loan_month_panel sorted by (loan_id, reporting_period).
               Requires: loan_id, reporting_period, delinquency_state,
               current_upb, origination_year.

    Returns:
        DataFrame: cohort_month, from_state, to_state,
                   loan_count, upb, transition_rate_pct
    """
    df = panel.sort(["loan_id", "reporting_period"])

    df = df.with_columns(pl.col("delinquency_state").shift(-1).over("loan_id").alias("next_state"))

    transitions = df.filter(
        pl.col("delinquency_state").is_not_null()
        & pl.col("next_state").is_not_null()
        & ~pl.col("delinquency_state").is_in(["UNKNOWN"])
    ).with_columns(pl.col("reporting_period").dt.truncate("1mo").alias("cohort_month"))

    agg = transitions.group_by(["cohort_month", "delinquency_state", "next_state"]).agg(
        [
            pl.len().alias("loan_count"),
            pl.col("current_upb").sum().alias("upb"),
        ]
    )

    # Compute transition rates: count(from→to) / count(from) per cohort
    totals = agg.group_by(["cohort_month", "delinquency_state"]).agg(
        pl.col("loan_count").sum().alias("from_total")
    )

    result = (
        agg.join(totals, on=["cohort_month", "delinquency_state"], how="left")
        .with_columns(
            (pl.col("loan_count") / pl.col("from_total") * 100).alias("transition_rate_pct")
        )
        .drop("from_total")
        .rename({"delinquency_state": "from_state", "next_state": "to_state"})
        .sort(["cohort_month", "from_state", "to_state"])
    )
    return result


def compute_roll_rates(transition_matrix: pl.DataFrame) -> pl.DataFrame:
    """Extract key transition rates by cohort month.

    Args:
        transition_matrix: Output of build_transition_matrix().

    Returns:
        DataFrame with one row per cohort_month and one column
        per key transition (e.g. new_delinquency_rate, cure_rate_30, ...).
    """
    result = transition_matrix.select("cohort_month").unique().sort("cohort_month")

    for from_state, to_state, col_name in KEY_TRANSITIONS:
        subset = (
            transition_matrix.filter(
                (pl.col("from_state") == from_state) & (pl.col("to_state") == to_state)
            )
            .select(["cohort_month", "transition_rate_pct"])
            .rename({"transition_rate_pct": col_name})
        )
        result = result.join(subset, on="cohort_month", how="left")

    return result


def delinquency_summary(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute monthly delinquency rates by reporting period.

    Returns:
        DataFrame with reporting_period, dq30_rate, dq60_rate,
        dq90plus_rate, serious_dq_rate (all as % of active UPB).
    """
    active = panel.filter(~pl.col("delinquency_state").is_in(["TERMINAL", "UNKNOWN"]))

    return (
        active.group_by("reporting_period")
        .agg(
            [
                pl.col("current_upb").sum().alias("total_upb"),
                pl.col("current_upb")
                .filter(pl.col("delinquency_state") == "DQ30")
                .sum()
                .alias("dq30_upb"),
                pl.col("current_upb")
                .filter(pl.col("delinquency_state") == "DQ60")
                .sum()
                .alias("dq60_upb"),
                pl.col("current_upb")
                .filter(pl.col("delinquency_state").is_in(["DQ90PLUS", "REO"]))
                .sum()
                .alias("dq90plus_upb"),
                pl.len().alias("loan_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("dq30_upb") / pl.col("total_upb") * 100).alias("dq30_rate"),
                (pl.col("dq60_upb") / pl.col("total_upb") * 100).alias("dq60_rate"),
                (pl.col("dq90plus_upb") / pl.col("total_upb") * 100).alias("dq90plus_rate"),
                ((pl.col("dq60_upb") + pl.col("dq90plus_upb")) / pl.col("total_upb") * 100).alias(
                    "serious_dq_rate"
                ),
            ]
        )
        .sort("reporting_period")
    )
