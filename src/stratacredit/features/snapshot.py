"""Leakage-safe model snapshot builder.

For each scoring date t, constructs a feature matrix where:
  - All features contain only information available at or before t
  - Targets look forward from t+1 through t+12

Leakage prevention is enforced via an explicit forbidden-column
list loaded from configs/features.yaml.

See docs/methodology.md for temporal validation design.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from stratacredit.config import features_config

# Columns that must NEVER appear as model features
_FORBIDDEN = set(
    features_config["leakage_guard"]["forbidden_always"]
    + features_config["leakage_guard"]["post_event_forbidden"]
)


def _assert_no_leakage(df: pl.DataFrame) -> None:
    """Raise if any forbidden columns are present in df."""
    leaking = _FORBIDDEN & set(df.columns)
    if leaking:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED: columns {sorted(leaking)} must not "
            f"appear in the feature matrix."
        )


def build_delinquency_history(
    panel: pl.DataFrame,
    as_of: date,
    lookback_months: int = 12,
) -> pl.DataFrame:
    """Build lagged delinquency history features as-of date t.

    For each loan active at t, computes:
      - months_in_30dpd_last_12m
      - months_in_60dpd_last_12m
      - months_in_90plus_last_12m
      - ever_modified
      - max_delinquency_ever
      - months_since_last_dq
      - prior_cure_events
    """
    cutoff = pl.lit(as_of)
    lookback = panel.filter(
        (pl.col("reporting_period") <= cutoff)
        & (pl.col("reporting_period") >= pl.lit(as_of).dt.offset_by(f"-{lookback_months}mo"))
    )

    hist = lookback.group_by("loan_id").agg(
        [
            pl.col("delinquency_months")
            .filter(pl.col("delinquency_months") == 1)
            .len()
            .alias("months_in_30dpd_last_12m"),
            pl.col("delinquency_months")
            .filter(pl.col("delinquency_months") == 2)
            .len()
            .alias("months_in_60dpd_last_12m"),
            pl.col("delinquency_months")
            .filter(pl.col("delinquency_months") >= 3)
            .len()
            .alias("months_in_90plus_last_12m"),
            pl.col("modification_flag").any().alias("ever_modified"),
            pl.col("delinquency_months").max().alias("max_delinquency_ever"),
        ]
    )

    # Months since last delinquency
    last_dq = (
        lookback.filter(pl.col("delinquency_months") > 0)
        .group_by("loan_id")
        .agg(pl.col("reporting_period").max().alias("last_dq_period"))
    )

    # Prior cure events: times loan went from DQ>0 back to CURRENT
    # Approximate as months where delinquency_months == 0 following DQ > 0
    cures = (
        lookback.sort(["loan_id", "reporting_period"])
        .with_columns(pl.col("delinquency_months").shift(1).over("loan_id").alias("prior_dq"))
        .filter((pl.col("delinquency_months") == 0) & (pl.col("prior_dq") > 0))
        .group_by("loan_id")
        .agg(pl.len().alias("prior_cure_events"))
    )

    result = (
        hist.join(last_dq, on="loan_id", how="left")
        .join(cures, on="loan_id", how="left")
        .with_columns(pl.col("prior_cure_events").fill_null(0))
    )
    return result


def build_snapshot(
    panel: pl.DataFrame,
    scoring_date: date,
    horizon_months: int = 12,
    min_forward_months: int = 12,
) -> pl.DataFrame:
    """Build a leakage-safe modeling snapshot at scoring_date.

    Args:
        panel: Full loan_month_panel sorted by (loan_id, reporting_period).
        scoring_date: The as-of date t. Features are from rows at t.
        horizon_months: Look-forward window for targets (default 12).
        min_forward_months: Minimum future months required for labeling.

    Returns:
        DataFrame with features + targets. One row per eligible loan.
        Target columns: credit_event_12m, serious_delinquency_12m,
        prepayment_12m. Rows without enough future data have null targets.
    """
    sd = pl.lit(scoring_date)
    panel["reporting_period"].max()

    # ── Features: panel row at scoring_date ──────────────────────────────────
    features = panel.filter(
        (pl.col("reporting_period") == sd) & (pl.col("is_terminal") == False)  # noqa: E712
    )

    if len(features) == 0:
        return pl.DataFrame()

    # Drop all forbidden/post-event columns from features
    drop_cols = [c for c in features.columns if c in _FORBIDDEN]
    features = features.drop(drop_cols)

    # ── Delinquency history features ─────────────────────────────────────────
    hist = build_delinquency_history(panel, scoring_date)
    features = features.join(hist, on="loan_id", how="left")

    # ── Targets: look forward t+1 through t+horizon ───────────────────────────
    horizon_end = pl.lit(scoring_date).dt.offset_by(f"{horizon_months}mo")

    forward = panel.filter(
        (pl.col("reporting_period") > sd) & (pl.col("reporting_period") <= horizon_end)
    )

    # credit_event_12m: any credit event in forward window
    ce_target = forward.group_by("loan_id").agg(
        pl.col("is_credit_event").any().alias("credit_event_12m"),
        pl.col("is_seriously_delinquent").any().alias("serious_delinquency_12m"),
        pl.col("is_voluntary_prepayment").any().alias("prepayment_12m"),
        pl.len().alias("_forward_months"),
    )

    result = (
        features.join(ce_target, on="loan_id", how="left")
        .with_columns(
            [
                # Null out targets if insufficient forward data
                pl.when(
                    pl.col("_forward_months").is_null()
                    | (pl.col("_forward_months") < min_forward_months)
                )
                .then(None)
                .otherwise(pl.col("credit_event_12m").cast(pl.Int8))
                .alias("credit_event_12m"),
                pl.when(
                    pl.col("_forward_months").is_null()
                    | (pl.col("_forward_months") < min_forward_months)
                )
                .then(None)
                .otherwise(pl.col("serious_delinquency_12m").cast(pl.Int8))
                .alias("serious_delinquency_12m"),
                pl.when(
                    pl.col("_forward_months").is_null()
                    | (pl.col("_forward_months") < min_forward_months)
                )
                .then(None)
                .otherwise(pl.col("prepayment_12m").cast(pl.Int8))
                .alias("prepayment_12m"),
            ]
        )
        .drop(["_forward_months"])
        .with_columns(pl.lit(str(scoring_date)).alias("scoring_date"))
    )

    # Final leakage assertion
    target_cols = {"credit_event_12m", "serious_delinquency_12m", "prepayment_12m", "scoring_date"}
    feature_df = result.drop([c for c in target_cols if c in result.columns])
    _assert_no_leakage(feature_df)

    return result


def build_training_dataset(
    panel: pl.DataFrame,
    train_vintage_start: int,
    train_vintage_end: int,
    horizon_months: int = 12,
    sample_frequency: str = "quarterly",
    max_scoring_dates: int | None = 12,
) -> pl.DataFrame:
    """Build a full training dataset from multiple scoring dates.

    Samples scoring dates within the training vintage window,
    then stacks snapshots into one training DataFrame.

    Args:
        panel: Full loan_month_panel.
        train_vintage_start: First origination year to include.
        train_vintage_end: Last origination year to include.
        horizon_months: Look-forward horizon.
        sample_frequency: 'monthly' or 'quarterly'.
        max_scoring_dates: Cap evenly distributed snapshot dates to bound
            memory for production-scale panels. ``None`` keeps every date.

    Returns:
        Stacked training DataFrame.
    """
    # Filter to training vintages
    train_panel = panel.filter(
        pl.col("origination_year").is_between(train_vintage_start, train_vintage_end)
    )

    # Get all reporting periods
    periods = sorted(train_panel["reporting_period"].unique().to_list())

    # Sample at desired frequency
    if sample_frequency == "quarterly":
        periods = [p for i, p in enumerate(periods) if i % 3 == 0]

    if max_scoring_dates and len(periods) > max_scoring_dates:
        # Retain a deterministic, evenly distributed history rather than the
        # earliest dates only; this avoids repeatedly materialising dozens of
        # near-identical 12-month look-back windows on large panels.
        last = len(periods) - 1
        indices = sorted(
            {round(i * last / (max_scoring_dates - 1)) for i in range(max_scoring_dates)}
        )
        periods = [periods[i] for i in indices]

    snapshots = []
    for period in periods:
        snap = build_snapshot(
            train_panel,
            scoring_date=period,
            horizon_months=horizon_months,
            min_forward_months=horizon_months,
        )
        if len(snap) > 0:
            snapshots.append(snap)

    if not snapshots:
        return pl.DataFrame()

    return pl.concat(snapshots, how="diagonal")
