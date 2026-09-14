"""XGBoost nonlinear models for credit events and prepayment."""

from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.impute import SimpleImputer

from stratacredit.features.encoder import encode_categoricals, get_feature_columns, to_numpy
from stratacredit.models.base import ClassificationMetrics, evaluate_classification

_DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
}


def train_xgb(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
    params: dict | None = None,
) -> tuple[xgb.XGBClassifier, list[str], ClassificationMetrics, ClassificationMetrics]:
    """Train an XGBoost binary classifier.

    Args:
        train_df: Training snapshot DataFrame.
        val_df: Validation snapshot DataFrame (for early stopping).
        target_col: Binary target column name.
        feature_cols: Feature columns. Auto-detected if None.
        n_estimators: Maximum boosting rounds.
        early_stopping_rounds: Rounds without improvement to stop.
        params: Override default XGBoost parameters.

    Returns:
        (model, feature_cols, train_metrics, val_metrics)
    """
    p = {**_DEFAULT_PARAMS, **(params or {})}

    train_df = encode_categoricals(train_df).filter(pl.col(target_col).is_not_null())
    val_df = encode_categoricals(val_df).filter(pl.col(target_col).is_not_null())

    if feature_cols is None:
        feature_cols = get_feature_columns(train_df, target_cols=[target_col])

    imputer = SimpleImputer(strategy="median")
    X_train, y_train = to_numpy(train_df, feature_cols, target_col)
    X_val, y_val = to_numpy(val_df, feature_cols, target_col)

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    # Handle class imbalance
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    p["scale_pos_weight"] = pos_weight

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="aucpr",
        **p,
    )
    model.fit(
        X_train,
        y_train.astype(int),
        eval_set=[(X_val, y_val.astype(int))],
        verbose=False,
    )
    # Store imputer on model for later use
    model._imputer = imputer
    model._feature_cols = feature_cols

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores = model.predict_proba(X_val)[:, 1]
    train_m = evaluate_classification(y_train, train_scores, split="train")
    val_m = evaluate_classification(y_val, val_scores, split="validation")

    return model, feature_cols, train_m, val_m


def score_xgb(
    model: xgb.XGBClassifier,
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
    split: str = "",
) -> tuple[np.ndarray, ClassificationMetrics | None]:
    """Score a fitted XGBoost model."""
    df_enc = encode_categoricals(df)
    X, y = to_numpy(df_enc, feature_cols, target_col)
    imputer = getattr(model, "_imputer", None)
    if imputer is not None:
        X = imputer.transform(X)
    scores = model.predict_proba(X)[:, 1]
    metrics = None
    if y is not None:
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            metrics = evaluate_classification(y[valid], scores[valid], split=split)
    return scores, metrics


def feature_importance_xgb(
    model: xgb.XGBClassifier,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Extract XGBoost feature importance scores."""
    scores = model.feature_importances_
    return pl.DataFrame({"feature": feature_cols, "importance": scores.tolist()}).sort(
        "importance", descending=True
    )
