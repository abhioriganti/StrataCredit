"""Stratification engine.

Produces collateral stratification tables by FICO, LTV, DTI,
coupon, vintage, state, occupancy, loan purpose, property type,
loan age, and delinquency status.

Each stratification returns a DataFrame with:
    band / category, loan_count, current_upb, pool_pct,
    wa_coupon, wa_fico, wa_ltv, wa_dti
"""

from __future__ import annotations

import polars as pl

# ── Band boundary definitions ─────────────────────────────────────────────────

FICO_BANDS = [
    (300, 620, "< 620"),
    (620, 640, "620–639"),
    (640, 660, "640–659"),
    (660, 680, "660–679"),
    (680, 700, "680–699"),
    (700, 720, "700–719"),
    (720, 740, "720–739"),
    (740, 760, "740–759"),
    (760, 780, "760–779"),
    (780, 851, "780+"),
]

LTV_BANDS = [
    (0, 60, "≤ 60%"),
    (60, 65, "60–65%"),
    (65, 70, "65–70%"),
    (70, 75, "70–75%"),
    (75, 80, "75–80%"),
    (80, 85, "80–85%"),
    (85, 90, "85–90%"),
    (90, 95, "90–95%"),
    (95, 200, "95%+"),
]

DTI_BANDS = [
    (0, 28, "≤ 28%"),
    (28, 36, "28–36%"),
    (36, 43, "36–43%"),
    (43, 50, "43–50%"),
    (50, 100, "50%+"),
]

COUPON_BANDS = [
    (0.0, 3.0, "< 3.0%"),
    (3.0, 3.5, "3.0–3.5%"),
    (3.5, 4.0, "3.5–4.0%"),
    (4.0, 4.5, "4.0–4.5%"),
    (4.5, 5.0, "4.5–5.0%"),
    (5.0, 5.5, "5.0–5.5%"),
    (5.5, 6.0, "5.5–6.0%"),
    (6.0, 7.0, "6.0–7.0%"),
    (7.0, 20.0, "7.0%+"),
]

LOAN_AGE_BANDS = [
    (0, 12, "0–12 mo"),
    (12, 24, "12–24 mo"),
    (24, 36, "24–36 mo"),
    (36, 60, "36–60 mo"),
    (60, 84, "60–84 mo"),
    (84, 120, "84–120 mo"),
    (120, 999, "120+ mo"),
]


def _assign_band(df: pl.DataFrame, col: str, bands: list[tuple]) -> pl.Series:
    """Assign band labels to a numeric column."""
    expr = pl.lit("Other")
    for lo, hi, label in reversed(bands):
        expr = pl.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(label)).otherwise(expr)
    return df.select(expr.alias("band"))["band"]


def _strat_summary(
    df: pl.DataFrame,
    group_col: str,
    total_upb: float,
) -> pl.DataFrame:
    """Aggregate a DataFrame by group_col into a stratification table."""
    agg = (
        df.group_by(group_col)
        .agg(
            [
                pl.len().alias("loan_count"),
                pl.col("current_upb").sum().alias("current_upb"),
                pl.col("current_upb").sum().alias("_upb_w"),
                # UPB-weighted averages via sum of products
                (pl.col("current_interest_rate") * pl.col("current_upb")).sum().alias("_wac_num"),
                (pl.col("fico").fill_null(0) * pl.col("current_upb")).sum().alias("_fico_num"),
                pl.col("fico")
                .is_not_null()
                .cast(pl.Float64)
                .mul(pl.col("current_upb"))
                .sum()
                .alias("_fico_w"),
                (pl.col("original_ltv").fill_null(0) * pl.col("current_upb"))
                .sum()
                .alias("_ltv_num"),
                pl.col("original_ltv")
                .is_not_null()
                .cast(pl.Float64)
                .mul(pl.col("current_upb"))
                .sum()
                .alias("_ltv_w"),
                (pl.col("dti").fill_null(0) * pl.col("current_upb")).sum().alias("_dti_num"),
                pl.col("dti")
                .is_not_null()
                .cast(pl.Float64)
                .mul(pl.col("current_upb"))
                .sum()
                .alias("_dti_w"),
            ]
        )
        .with_columns(
            [
                (pl.col("current_upb") / total_upb * 100).alias("pool_pct"),
                (pl.col("_wac_num") / pl.col("_upb_w")).alias("wa_coupon"),
                (pl.col("_fico_num") / pl.col("_fico_w")).alias("wa_fico"),
                (pl.col("_ltv_num") / pl.col("_ltv_w")).alias("wa_ltv"),
                (pl.col("_dti_num") / pl.col("_dti_w")).alias("wa_dti"),
            ]
        )
        .drop(
            [
                "_upb_w",
                "_wac_num",
                "_fico_num",
                "_fico_w",
                "_ltv_num",
                "_ltv_w",
                "_dti_num",
                "_dti_w",
            ]
        )
        .sort(group_col)
    )
    return agg


def stratify_by_band(
    df: pl.DataFrame,
    col: str,
    bands: list[tuple],
    band_label: str = "band",
) -> pl.DataFrame:
    """Stratify panel by a numeric band column."""
    total_upb = float(df["current_upb"].sum() or 1.0)
    df = df.with_columns(_assign_band(df, col, bands).alias(band_label))
    return _strat_summary(df, band_label, total_upb)


def stratify_by_category(
    df: pl.DataFrame,
    col: str,
) -> pl.DataFrame:
    """Stratify panel by a categorical column."""
    total_upb = float(df["current_upb"].sum() or 1.0)
    return _strat_summary(df, col, total_upb)


# ── Public stratification functions ───────────────────────────────────────────


def strat_fico(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_band(df, "fico", FICO_BANDS, "fico_band")


def strat_ltv(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_band(df, "original_ltv", LTV_BANDS, "ltv_band")


def strat_dti(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_band(df, "dti", DTI_BANDS, "dti_band")


def strat_coupon(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_band(df, "current_interest_rate", COUPON_BANDS, "coupon_band")


def strat_loan_age(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_band(df, "loan_age", LOAN_AGE_BANDS, "loan_age_band")


def strat_vintage(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_category(df, "origination_year")


def strat_state(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_category(df, "property_state")


def strat_loan_purpose(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_category(df, "loan_purpose")


def strat_occupancy(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_category(df, "occupancy")


def strat_property_type(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_category(df, "property_type")


def strat_delinquency(df: pl.DataFrame) -> pl.DataFrame:
    return stratify_by_category(df, "delinquency_state")


ALL_STRATIFICATIONS = {
    "fico": strat_fico,
    "ltv": strat_ltv,
    "dti": strat_dti,
    "coupon": strat_coupon,
    "loan_age": strat_loan_age,
    "vintage": strat_vintage,
    "state": strat_state,
    "loan_purpose": strat_loan_purpose,
    "occupancy": strat_occupancy,
    "property_type": strat_property_type,
    "delinquency": strat_delinquency,
}
