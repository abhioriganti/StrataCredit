"""Reusable interpretable logistic-regression baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stratacredit.features.encoder import encode_categoricals, get_feature_columns, to_numpy
from stratacredit.models.evaluation import classification_metrics


@dataclass
class LogisticMetrics:
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    n_observations: int
    n_positive: int


def train_logistic(df: pl.DataFrame, target: str) -> tuple[Pipeline, list[str], LogisticMetrics]:
    """Train an interpretable baseline using only typed, leakage-safe columns."""
    df = encode_categoricals(df).drop_nulls([target])
    features = get_feature_columns(df, [target, "credit_event_12m", "prepayment_12m", "serious_delinquency_12m"])
    x, y = to_numpy(df, features, target)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{target} requires both event and non-event observations")
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    pipeline.fit(x, y)
    p = pipeline.predict_proba(x)[:, 1]
    values = classification_metrics(y, p)
    return pipeline, features, LogisticMetrics(values.get("roc_auc", float("nan")), values.get("pr_auc", float("nan")), values["brier_score"], values["log_loss"], len(y), int(y.sum()))


def score_logistic(pipeline: Pipeline, df: pl.DataFrame, feature_columns: list[str], target: str | None = None, split: str = "score") -> tuple[pl.DataFrame, LogisticMetrics | None]:
    """Score observations and, where labels exist, return diagnostics."""
    scored = encode_categoricals(df)
    missing = [c for c in feature_columns if c not in scored.columns]
    if missing:
        scored = scored.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in missing])
    x, y = to_numpy(scored, feature_columns, target)
    probability = pipeline.predict_proba(x)[:, 1]
    scored = scored.with_columns(pl.Series(f"predicted_{target or split}", probability))
    if y is None or np.isnan(y).any() or len(np.unique(y)) < 2:
        return scored, None
    values = classification_metrics(y, probability)
    return scored, LogisticMetrics(values["roc_auc"], values["pr_auc"], values["brier_score"], values["log_loss"], len(y), int(y.sum()))
