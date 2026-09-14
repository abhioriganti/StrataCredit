"""Loss severity regression model.

Predicts realized loss severity conditional on a credit event.
Features are origination characteristics + delinquency history.
Post-event fields (recoveries, sale proceeds) are never used.

Implements:
  - Ridge regression baseline
  - XGBoost regression
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stratacredit.features.encoder import encode_categoricals, get_feature_columns, to_numpy
from stratacredit.models.base import SeverityMetrics, evaluate_severity
from stratacredit.performance.severity import compute_severity

# Post-event columns that must never be features for severity model
_SEV_FORBIDDEN = {
    "net_sale_proceeds",
    "mi_recoveries",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_costs",
    "taxes_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "modification_cost",
    "delinquent_accrued_interest",
    "total_recoveries",
    "recovery_rate",
    "over_recovery_flag",
    "loss_severity",
}


def prepare_severity_data(panel: pl.DataFrame) -> pl.DataFrame:
    """Extract severity training data from the panel.

    Returns one row per credit-event loan with loss_severity as target.
    Drops post-event fields and retains origination + delinquency features.
    """
    sev = compute_severity(panel)
    sev = sev.filter(
        pl.col("loss_severity").is_not_null()
        & pl.col("loss_severity").is_finite()
        # Keep losses in a reasonable range [0, 1.5]
        & (pl.col("loss_severity") >= 0)
        & (pl.col("loss_severity") <= 1.5)
    )
    # Drop post-event columns
    drop = [c for c in sev.columns if c in _SEV_FORBIDDEN - {"loss_severity"}]
    return sev.drop(drop)


def train_severity_ridge(
    train_df: pl.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[Pipeline, list[str], SeverityMetrics]:
    """Train a Ridge regression severity model."""
    df_enc = encode_categoricals(train_df)
    if feature_cols is None:
        feature_cols = get_feature_columns(df_enc, target_cols=["loss_severity"])
        feature_cols = [c for c in feature_cols if c not in _SEV_FORBIDDEN]

    X, y = to_numpy(df_enc, feature_cols, "loss_severity")
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    metrics = evaluate_severity(y, y_pred, split="train")
    return pipe, feature_cols, metrics


def train_severity_xgb(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    n_estimators: int = 300,
) -> tuple[xgb.XGBRegressor, list[str], SimpleImputer, SeverityMetrics, SeverityMetrics]:
    """Train an XGBoost severity regression model."""
    params = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
    }

    train_enc = encode_categoricals(train_df)
    val_enc = encode_categoricals(val_df)

    if feature_cols is None:
        feature_cols = get_feature_columns(train_enc, target_cols=["loss_severity"])
        feature_cols = [c for c in feature_cols if c not in _SEV_FORBIDDEN]

    X_tr, y_tr = to_numpy(train_enc, feature_cols, "loss_severity")
    X_va, y_va = to_numpy(val_enc, feature_cols, "loss_severity")

    mask_tr = ~np.isnan(y_tr)
    mask_va = ~np.isnan(y_va)
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_va, y_va = X_va[mask_va], y_va[mask_va]

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    X_va = imputer.transform(X_va)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=30,
        eval_metric="mae",
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    model._imputer = imputer

    tr_m = evaluate_severity(y_tr, model.predict(X_tr), split="train")
    va_m = evaluate_severity(y_va, model.predict(X_va), split="validation")
    return model, feature_cols, imputer, tr_m, va_m


def predict_severity(model, df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Predict severity for a set of loans."""
    df_enc = encode_categoricals(df)
    X, _ = to_numpy(df_enc, feature_cols)
    imputer = getattr(model, "_imputer", None)
    if imputer is not None:
        X = imputer.transform(X)
    preds = model.predict(X)
    return np.clip(preds, 0.0, 1.5)
