"""Feature encoding utilities.

Encodes categorical variables and prepares the feature matrix
for scikit-learn / XGBoost consumption.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Categorical mappings for known codes
LOAN_PURPOSE_MAP = {"P": 0, "C": 1, "R": 2, "U": 3}
OCCUPANCY_MAP = {"P": 0, "S": 1, "I": 2, "U": 3}
PROPERTY_TYPE_MAP = {"SF": 0, "CO": 1, "MH": 2, "CP": 3, "PU": 4}
CHANNEL_MAP = {"R": 0, "B": 1, "C": 2, "T": 3}


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """Encode categorical columns to integers.

    Applies known code mappings; unknown values map to -1.
    Binary Y/N flags are mapped to 1/0.
    """
    exprs = []

    for col, mapping in [
        ("loan_purpose", LOAN_PURPOSE_MAP),
        ("occupancy", OCCUPANCY_MAP),
        ("property_type", PROPERTY_TYPE_MAP),
        ("channel", CHANNEL_MAP),
    ]:
        if col in df.columns:
            exprs.append(
                pl.col(col).replace(mapping, default=-1).cast(pl.Int8).alias(f"{col}_encoded")
            )

    for flag_col in [
        "first_time_homebuyer",
        "modification_flag",
        "interest_only",
        "super_conforming",
        "prepayment_penalty",
    ]:
        if flag_col in df.columns:
            exprs.append(pl.col(flag_col).cast(pl.Int8).alias(f"{flag_col}_flag"))

    if exprs:
        df = df.with_columns(exprs)
    return df


def get_feature_columns(df: pl.DataFrame, target_cols: list[str]) -> list[str]:
    """Return numeric feature columns, excluding targets and metadata."""
    exclude = set(target_cols) | {
        "loan_id",
        "scoring_date",
        "reporting_period",
        "origination_date",
        "first_payment_date",
        "maturity_date",
        "zero_balance_date",
        "_source_file",
        "_source_vintage",
        "_ingested_at",
        "_file_checksum",
        # String categoricals (use encoded versions instead)
        "loan_purpose",
        "occupancy",
        "property_type",
        "channel",
        "property_state",
        "msa",
        "postal_code",
        "delinquency_state",
        "raw_delinquency_status",
        "terminal_event_type",
        "zero_balance_code",
        "amort_type",
        "program_indicator",
        "borrower_assistance_status_code",
        "seller_name",
        "servicer_name",
        "vintage",
    }
    return [
        c
        for c in df.columns
        if c not in exclude
        and df[c].dtype
        in (
            pl.Float64,
            pl.Float32,
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.Boolean,
        )
    ]


def to_numpy(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a Polars DataFrame to numpy arrays for sklearn/XGBoost.

    Args:
        df: Feature DataFrame.
        feature_cols: List of feature column names.
        target_col: Optional target column name.

    Returns:
        (X, y) where y is None if target_col is not provided.
    """
    X = df.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)

    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].cast(pl.Float32).to_numpy()

    return X, y
