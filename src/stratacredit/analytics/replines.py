"""Repline generator.

Aggregates loans into representative collateral cells (replines)
grouped by FICO × LTV × coupon × vintage × state_group.

Replines are the standard output format for securitization
collateral submissions.
"""

from __future__ import annotations

import polars as pl

from stratacredit.analytics.stratification import (
    COUPON_BANDS,
    FICO_BANDS,
    LTV_BANDS,
    _assign_band,
)

STATE_GROUPS = {
    "Northeast": ["CT", "MA", "ME", "NH", "NJ", "NY", "PA", "RI", "VT"],
    "Southeast": ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "VA", "WV"],
    "Midwest": ["IA", "IL", "IN", "KS", "MI", "MN", "MO", "ND", "NE", "OH", "SD", "WI"],
    "Southwest": ["AZ", "NM", "OK", "TX"],
    "West": ["AK", "CA", "CO", "HI", "ID", "MT", "NV", "OR", "UT", "WA", "WY"],
    "DC_MD": ["DC", "MD"],
}

_STATE_MAP = {state: region for region, states in STATE_GROUPS.items() for state in states}


def _assign_state_group(df: pl.DataFrame) -> pl.Series:
    """Map property_state to regional group."""
    if "property_state" not in df.columns:
        return pl.Series(["Other"] * len(df))
    return df["property_state"].replace(_STATE_MAP, default="Other")


def build_replines(
    df: pl.DataFrame,
    min_cell_loans: int = 5,
) -> pl.DataFrame:
    """Build repline table from a panel snapshot.

    Args:
        df: Active loan panel snapshot (one reporting period).
        min_cell_loans: Minimum loans per repline cell.
            Cells with fewer loans are included but flagged.

    Returns:
        Polars DataFrame of repline cells with aggregated metrics.
    """
    total_upb = float(df["current_upb"].sum() or 1.0)

    # Assign band labels
    df = df.with_columns(
        [
            _assign_band(df, "fico", FICO_BANDS).alias("fico_band"),
            _assign_band(df, "original_ltv", LTV_BANDS).alias("ltv_band"),
            _assign_band(df, "original_interest_rate", COUPON_BANDS).alias("coupon_band"),
            _assign_state_group(df).alias("state_group"),
        ]
    )

    if "origination_year" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("origination_year"))

    group_cols = ["fico_band", "ltv_band", "coupon_band", "origination_year", "state_group"]

    replines = (
        df.group_by(group_cols)
        .agg(
            [
                pl.len().alias("loan_count"),
                pl.col("current_upb").sum().alias("upb"),
                # Weighted averages
                (pl.col("current_interest_rate") * pl.col("current_upb")).sum().alias("_wac_n"),
                pl.col("current_upb").sum().alias("_w"),
                (pl.col("fico").fill_null(0) * pl.col("current_upb")).sum().alias("_fico_n"),
                pl.col("fico")
                .is_not_null()
                .cast(pl.Float64)
                .mul(pl.col("current_upb"))
                .sum()
                .alias("_fico_w"),
                (pl.col("original_ltv").fill_null(0) * pl.col("current_upb")).sum().alias("_ltv_n"),
                pl.col("original_ltv")
                .is_not_null()
                .cast(pl.Float64)
                .mul(pl.col("current_upb"))
                .sum()
                .alias("_ltv_w"),
                (pl.col("dti").fill_null(0) * pl.col("current_upb")).sum().alias("_dti_n"),
                pl.col("dti")
                .is_not_null()
                .cast(pl.Float64)
                .mul(pl.col("current_upb"))
                .sum()
                .alias("_dti_w"),
                (pl.col("loan_age").fill_null(0) * pl.col("current_upb")).sum().alias("_age_n"),
            ]
        )
        .with_columns(
            [
                (pl.col("upb") / total_upb * 100).alias("pool_pct"),
                (pl.col("_wac_n") / pl.col("_w")).alias("wac"),
                (pl.col("_fico_n") / pl.col("_fico_w")).alias("wa_fico"),
                (pl.col("_ltv_n") / pl.col("_ltv_w")).alias("wa_ltv"),
                (pl.col("_dti_n") / pl.col("_dti_w")).alias("wa_dti"),
                (pl.col("_age_n") / pl.col("_w")).alias("wala"),
                (pl.col("loan_count") < min_cell_loans).alias("small_cell"),
            ]
        )
        .drop(
            ["_wac_n", "_w", "_fico_n", "_fico_w", "_ltv_n", "_ltv_w", "_dti_n", "_dti_w", "_age_n"]
        )
        .sort(["origination_year", "fico_band", "ltv_band", "coupon_band", "state_group"])
    )

    return replines
