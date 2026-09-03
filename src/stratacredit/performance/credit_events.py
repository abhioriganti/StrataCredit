"""Credit event and default analytics.

Definitions
-----------
serious_delinquency:
    Loan reaches 90+ DPD at any point.

terminal_credit_event:
    Loan terminates via a non-voluntary zero balance code:
      '02' third_party_sale
      '03' short_sale
      '09' reo_disposition
      '15' note_sale

voluntary_prepayment:
    Loan terminates via zero balance code '01'.

CDR (Conditional Default Rate):
    Monthly annualised rate of new credit events among active loans.

    CDR_t = 1 - (1 - (new_credit_events_t / active_loans_{t-1}))^12

    This is StrataCredit's internal CDR definition.
    It does NOT replicate any rating-agency methodology.

See docs/methodology.md for full definitions.
"""

from __future__ import annotations

import polars as pl

CREDIT_EVENT_ZBC = ["02", "03", "09", "15"]
VOLUNTARY_PREPAY_ZBC = ["01"]


def classify_terminations(panel: pl.DataFrame) -> pl.DataFrame:
    """Extract terminal event rows with classification.

    Args:
        panel: Full loan_month_panel.

    Returns:
        DataFrame of one row per terminated loan with:
        loan_id, zero_balance_date, terminal_event_type,
        origination_year, loan_age, fico, original_ltv, ...
    """
    return (
        panel.filter(pl.col("is_terminal") == True)  # noqa: E712
        .select(
            [
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
                    "current_upb",
                    "zero_balance_removal_upb",
                    "actual_loss",
                    "is_credit_event",
                    "is_voluntary_prepayment",
                ]
                if c in panel.columns
            ]
        )
        .unique(subset=["loan_id"])
    )


def monthly_credit_event_rate(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute monthly credit event rate and CDR.

    Returns:
        DataFrame with reporting_period, active_loans, new_credit_events,
        monthly_rate, cdr (annualised).
    """
    # Active loans at start of each period
    active = (
        panel.filter(pl.col("delinquency_state") != "TERMINAL")
        .group_by("reporting_period")
        .agg(pl.len().alias("active_loans"))
        .sort("reporting_period")
    )

    # New credit events per period
    events = (
        panel.filter(pl.col("is_credit_event") == True)  # noqa: E712
        .group_by("reporting_period")
        .agg(pl.len().alias("new_credit_events"))
    )

    result = (
        active.join(events, on="reporting_period", how="left")
        .with_columns(pl.col("new_credit_events").fill_null(0))
        .with_columns((pl.col("new_credit_events") / pl.col("active_loans")).alias("monthly_rate"))
        .with_columns(
            # CDR = 1 - (1 - monthly_rate)^12
            ((1.0 - (1.0 - pl.col("monthly_rate")).pow(12)) * 100.0).alias("cdr")
        )
        .sort("reporting_period")
    )
    return result


def vintage_credit_event_matrix(panel: pl.DataFrame) -> pl.DataFrame:
    """Build vintage × months-on-book cumulative credit event matrix.

    Returns:
        DataFrame with origination_year, loan_age_bucket,
        cumulative_credit_event_rate (% of original loan count).
    """
    orig_counts = panel.group_by("origination_year").agg(
        pl.col("loan_id").n_unique().alias("orig_loan_count")
    )

    events_by_vintage_age = (
        panel.filter(pl.col("is_credit_event") == True)  # noqa: E712
        .group_by(["origination_year", "loan_age"])
        .agg(pl.len().alias("credit_events"))
        .sort(["origination_year", "loan_age"])
    )

    # Cumulative sum per vintage
    events_cum = (
        events_by_vintage_age.with_columns(
            pl.col("credit_events").cum_sum().over("origination_year").alias("cumulative_events")
        )
        .join(orig_counts, on="origination_year", how="left")
        .with_columns(
            (pl.col("cumulative_events") / pl.col("orig_loan_count") * 100).alias(
                "cumulative_credit_event_rate"
            )
        )
    )
    return events_cum
