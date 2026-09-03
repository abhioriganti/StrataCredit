"""Loss severity analytics.

Loss Severity Definition
------------------------
    loss_severity = actual_loss / zero_balance_removal_upb

where:
    actual_loss = from Freddie Mac actual_loss_calculation field
    zero_balance_removal_upb = UPB at time of termination

Only computed for terminal credit-event loans (ZBC 02, 03, 09, 15)
that have both actual_loss and zero_balance_removal_upb populated.

Negative severities (recoveries exceeding UPB) are retained and
documented but flagged.  They represent genuine MI/sale proceeds
exceeding outstanding balance and should not be silently removed.

See docs/methodology.md for full recovery waterfall.
"""

from __future__ import annotations

import polars as pl


def compute_severity(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute loss severity for terminated credit-event loans.

    Args:
        panel: Full loan_month_panel.

    Returns:
        DataFrame with one row per credit-event termination and
        columns: loan_id, loss_severity, recovery_rate, plus
        all origination and termination fields.
    """
    credit_events = panel.filter(
        (pl.col("is_credit_event") == True)  # noqa: E712
        & pl.col("zero_balance_removal_upb").is_not_null()
        & (pl.col("zero_balance_removal_upb") > 0)
    )

    keep = [
        c
        for c in [
            "loan_id",
            "zero_balance_date",
            "terminal_event_type",
            "zero_balance_code",
            "origination_year",
            "loan_age",
            "fico",
            "original_ltv",
            "original_cltv",
            "dti",
            "property_state",
            "occupancy",
            "loan_purpose",
            "mortgage_insurance_pct",
            "zero_balance_removal_upb",
            "net_sale_proceeds",
            "mi_recoveries",
            "non_mi_recoveries",
            "expenses",
            "delinquent_accrued_interest",
            "actual_loss",
        ]
        if c in panel.columns
    ]

    sev = (
        credit_events.select(keep)
        .unique(subset=["loan_id"])
        .with_columns(
            [
                # Loss severity: actual_loss / termination_upb
                pl.when(
                    pl.col("zero_balance_removal_upb").is_not_null()
                    & (pl.col("zero_balance_removal_upb") > 0)
                    & pl.col("actual_loss").is_not_null()
                )
                .then(pl.col("actual_loss") / pl.col("zero_balance_removal_upb"))
                .otherwise(None)
                .alias("loss_severity"),
                # Total recoveries
                (
                    pl.col("mi_recoveries").fill_null(0.0)
                    + pl.col("non_mi_recoveries").fill_null(0.0)
                ).alias("total_recoveries"),
            ]
        )
        .with_columns(
            [
                # Recovery rate = total_recoveries / termination_upb
                pl.when(pl.col("zero_balance_removal_upb") > 0)
                .then(pl.col("total_recoveries") / pl.col("zero_balance_removal_upb"))
                .otherwise(None)
                .alias("recovery_rate"),
                # Flag negative severities (over-recovery)
                (pl.col("loss_severity") < 0).alias("over_recovery_flag"),
            ]
        )
    )
    return sev


def severity_summary(sev: pl.DataFrame) -> dict:
    """Compute aggregate severity statistics.

    Args:
        sev: Output of compute_severity().

    Returns:
        Dict with mean_severity, median_severity, wa_severity,
        mean_recovery_rate, count, over_recovery_count.
    """
    valid = sev.filter(pl.col("loss_severity").is_not_null())
    if len(valid) == 0:
        return {"count": 0}

    wa = None
    if "zero_balance_removal_upb" in valid.columns:
        num = (valid["loss_severity"] * valid["zero_balance_removal_upb"]).sum()
        den = valid["zero_balance_removal_upb"].sum()
        wa = float(num / den) if den else None

    return {
        "count": len(valid),
        "mean_severity": float(valid["loss_severity"].mean()),
        "median_severity": float(valid["loss_severity"].median()),
        "wa_severity": wa,
        "mean_recovery_rate": float(valid["recovery_rate"].mean())
        if "recovery_rate" in valid.columns
        else None,
        "over_recovery_count": int(valid["over_recovery_flag"].sum())
        if "over_recovery_flag" in valid.columns
        else 0,
    }


def severity_by_dimension(sev: pl.DataFrame, dim: str) -> pl.DataFrame:
    """Aggregate severity by a single dimension column.

    Args:
        sev: Output of compute_severity().
        dim: Column to group by (e.g. 'origination_year', 'fico_band').

    Returns:
        Aggregated DataFrame with mean and WA severity per group.
    """
    if dim not in sev.columns:
        raise ValueError(f"Column '{dim}' not in severity DataFrame")

    return (
        sev.filter(pl.col("loss_severity").is_not_null())
        .group_by(dim)
        .agg(
            [
                pl.len().alias("event_count"),
                pl.col("loss_severity").mean().alias("mean_severity"),
                pl.col("loss_severity").median().alias("median_severity"),
                (pl.col("loss_severity") * pl.col("zero_balance_removal_upb")).sum().alias("_n"),
                pl.col("zero_balance_removal_upb").sum().alias("_w"),
            ]
        )
        .with_columns((pl.col("_n") / pl.col("_w")).alias("wa_severity"))
        .drop(["_n", "_w"])
        .sort(dim)
    )
