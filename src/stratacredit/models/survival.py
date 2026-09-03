"""Survival analysis models.

Implements XGBoost AFT (Accelerated Failure Time) for:
  - time to voluntary prepayment
  - time to credit event

Right-censoring: loans that have not yet experienced the event
are censored at their last observed month.

The AFT model estimates the expected time-to-event; lower values
indicate higher near-term risk.

See docs/methodology.md for definitions.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.impute import SimpleImputer

from stratacredit.features.encoder import encode_categoricals, get_feature_columns, to_numpy


def build_survival_dataset(
    panel: pl.DataFrame,
    event_col: str,
    time_col: str = "loan_age",
) -> pl.DataFrame:
    """Build a survival dataset from the panel.

    For each loan, returns one row with:
      - time_lower: lower bound on event time (loan_age at event, or last obs)
      - time_upper: upper bound (-1 for censored, same as time_lower for events)
      - All feature columns

    Args:
        panel: loan_month_panel.
        event_col: e.g. 'is_credit_event' or 'is_voluntary_prepayment'.
        time_col: Time variable (default 'loan_age').

    Returns:
        One-row-per-loan survival DataFrame.
    """
    # Last observed row per loan
    last = (
        panel.sort(["loan_id", "reporting_period"])
        .group_by("loan_id")
        .agg([
            pl.col(event_col).any().alias("event_occurred"),
            pl.col(time_col).last().alias("time_obs"),
            # Feature columns: take last non-null value
            *[
                pl.col(c).last().alias(c)
                for c in panel.columns
                if c not in {"loan_id", event_col, time_col, "reporting_period"}
                and c not in {
                    "zero_balance_code", "zero_balance_effective_date",
                    "zero_balance_removal_upb", "actual_loss_calculation",
                    "net_sale_proceeds", "mi_recoveries", "non_mi_recoveries",
                    "expenses", "actual_loss",
                }
            ],
        ])
    )

    # For events: time_lower = time_upper = observed time
    # For censored: time_lower = observed time, time_upper = +inf (use -1 in XGB)
    last = last.with_columns([
        pl.col("time_obs").alias("time_lower"),
        pl.when(pl.col("event_occurred"))
        .then(pl.col("time_obs"))
        .otherwise(pl.lit(-1.0))   # -1 = right-censored in XGBoost AFT
        .alias("time_upper"),
    ])

    return last


def train_survival(
    train_df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
    n_estimators: int = 300,
) -> tuple[xgb.XGBRegressor, list[str], SimpleImputer]:
    """Train an XGBoost AFT survival model.

    Args:
        train_df: Output of build_survival_dataset().
        feature_cols: Feature columns. Auto-detected if None.
        params: Override XGBoost params.
        n_estimators: Boosting rounds.

    Returns:
        (model, feature_cols, imputer)
    """
    default_params = {
        "objective":           "survival:aft",
        "aft_loss_distribution": "normal",
        "max_depth":           5,
        "learning_rate":       0.05,
        "subsample":           0.8,
        "colsample_bytree":    0.8,
        "min_child_weight":    50,
        "tree_method":         "hist",
        "seed":                42,
        "verbosity":           0,
    }
    p = {**default_params, **(params or {})}

    df_enc = encode_categoricals(train_df)
    if feature_cols is None:
        feature_cols = get_feature_columns(
            df_enc,
            target_cols=["time_lower", "time_upper", "event_occurred", "time_obs"],
        )

    X, _ = to_numpy(df_enc, feature_cols)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    y_lower = train_df["time_lower"].to_numpy().astype(np.float32)
    y_upper = train_df["time_upper"].to_numpy().astype(np.float32)

    dtrain = xgb.DMatrix(X)
    dtrain.set_float_info("label_lower_bound", y_lower)
    dtrain.set_float_info("label_upper_bound", y_upper)

    model = xgb.train(
        p,
        dtrain,
        num_boost_round=n_estimators,
    )
    return model, feature_cols, imputer


def predict_survival(
    model: xgb.Booster,
    df: pl.DataFrame,
    feature_cols: list[str],
    imputer: SimpleImputer,
) -> np.ndarray:
    """Predict expected time-to-event for each loan.

    Returns:
        Array of predicted time-to-event (higher = lower near-term risk).
    """
    df_enc = encode_categoricals(df)
    X, _ = to_numpy(df_enc, feature_cols)
    X = imputer.transform(X)
    dmat = xgb.DMatrix(X)
    return model.predict(dmat)
