"""Prepayment analytics: SMM and CPR.

Methodology
-----------
For each non-delinquent loan in month t:

    scheduled_principal_t =
        UPB_{t-1} * (r/12) / ((1 + r/12)^n_{t-1} - 1)

    where r = prior month interest rate, n = prior remaining term.

    SMM_t = max(0, (UPB_{t-1} - UPB_t - scheduled_principal_t) / UPB_{t-1})

For loans that voluntarily prepaid in month t: SMM_t = 1.0

    CPR_t = 1 - (1 - SMM_t)^12

All CPRs are expressed as percentages (0–100).

See docs/methodology.md for full derivation.
"""

from __future__ import annotations

import polars as pl


def _scheduled_principal(
    prior_upb: pl.Series,
    prior_rate: pl.Series,
    prior_remaining_term: pl.Series,
) -> pl.Series:
    """Compute scheduled principal using standard amortization formula.

    P_sched = UPB * (r/12) / ((1 + r/12)^n - 1)
    """
    r = prior_rate / 1200.0  # monthly rate
    n = prior_remaining_term.cast(pl.Float64)
    factor = (1.0 + r).pow(n) - 1.0
    # Avoid division by zero for r=0 or n=0
    denom = pl.when(factor > 0).then(factor).otherwise(1.0)
    p_sched = prior_upb * r / denom
    # Zero out negatives and nulls
    return p_sched.fill_null(0.0).clip(lower_bound=0.0)


def compute_smm(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute SMM for each active loan-month row.

    Args:
        panel: Full loan_month_panel sorted by (loan_id, reporting_period).
               Must include current_upb, current_interest_rate, loan_age,
               remaining_term, is_voluntary_prepayment, delinquency_state.

    Returns:
        DataFrame with columns:
            loan_id, reporting_period, origination_year, loan_age,
            current_upb, smm, cpr
    """
    required = {
        "loan_id",
        "reporting_period",
        "current_upb",
        "current_interest_rate",
        "remaining_term",
        "is_voluntary_prepayment",
        "delinquency_state",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    df = panel.sort(["loan_id", "reporting_period"])

    # Lag UPB, rate, remaining term
    df = df.with_columns(
        [
            pl.col("current_upb").shift(1).over("loan_id").alias("prior_upb"),
            pl.col("current_interest_rate").shift(1).over("loan_id").alias("prior_rate"),
            pl.col("remaining_term").shift(1).over("loan_id").alias("prior_remaining_term"),
        ]
    )

    # Exclude severely delinquent loans from prepayment calculation
    active = df.filter(
        ~pl.col("delinquency_state").is_in(["DQ60", "DQ90PLUS", "REO", "TERMINAL"])
        & pl.col("prior_upb").is_not_null()
        & (pl.col("prior_upb") > 0)
    )

    # Scheduled principal
    sched_prin = _scheduled_principal(
        active["prior_upb"],
        active["prior_rate"].fill_null(0.0),
        active["prior_remaining_term"].fill_null(1.0),
    )

    # SMM
    smm = (
        pl.when(active["is_voluntary_prepayment"])
        .then(pl.lit(1.0))
        .otherwise(
            (
                (active["prior_upb"] - active["current_upb"].fill_null(0.0) - sched_prin)
                / active["prior_upb"]
            ).clip(lower_bound=0.0, upper_bound=1.0)
        )
    )

    # CPR = 1 - (1 - SMM)^12
    cpr = (1.0 - (1.0 - smm).pow(12)) * 100.0

    keep_cols = [
        c
        for c in [
            "loan_id",
            "reporting_period",
            "origination_year",
            "loan_age",
            "current_upb",
            "fico",
            "original_ltv",
            "loan_purpose",
            "property_state",
            "occupancy",
            "current_interest_rate",
            "is_voluntary_prepayment",
        ]
        if c in active.columns
    ]

    result = active.select(keep_cols).with_columns(
        [
            smm.alias("smm"),
            cpr.alias("cpr"),
        ]
    )
    return result


def aggregate_cpr(
    smm_df: pl.DataFrame,
    group_by: list[str] | None = None,
) -> pl.DataFrame:
    """Aggregate SMM/CPR to reporting-period level (or other grouping).

    Args:
        smm_df: Output of compute_smm().
        group_by: Columns to group by. Defaults to [reporting_period].

    Returns:
        Aggregated DataFrame with wa_smm, cpr_wa, loan_count, total_upb.
    """
    if group_by is None:
        group_by = ["reporting_period"]

    agg = (
        smm_df.group_by(group_by)
        .agg(
            [
                pl.len().alias("loan_count"),
                pl.col("current_upb").sum().alias("total_upb"),
                pl.col("smm").mean().alias("avg_smm"),
                # UPB-weighted SMM
                (pl.col("smm") * pl.col("current_upb")).sum().alias("_smm_n"),
                pl.col("current_upb").sum().alias("_w"),
            ]
        )
        .with_columns(
            [
                (pl.col("_smm_n") / pl.col("_w")).alias("wa_smm"),
            ]
        )
        .with_columns(
            [
                ((1.0 - (1.0 - pl.col("wa_smm")).pow(12)) * 100.0).alias("cpr_wa"),
            ]
        )
        .drop(["_smm_n", "_w"])
        .sort(group_by)
    )
    return agg
