"""Shared model utilities and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss_val: float
    top_decile_capture: float
    n_positive: int
    n_total: int
    split: str = ""


@dataclass
class SeverityMetrics:
    mae: float
    rmse: float
    bias: float
    balance_weighted_mae: float | None = None
    split: str = ""


def evaluate_classification(
    y_true: np.ndarray,
    y_score: np.ndarray,
    split: str = "",
    weights: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute standard binary classification metrics."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]
    w = weights[valid] if weights is not None else None

    roc = float(roc_auc_score(y_true, y_score, sample_weight=w))
    prauc = float(average_precision_score(y_true, y_score, sample_weight=w))
    brier = float(brier_score_loss(y_true, y_score, sample_weight=w))
    ll = float(log_loss(y_true, y_score, sample_weight=w))

    # Top-decile capture: share of positives in top 10% of scores
    n = len(y_score)
    top_n = max(1, n // 10)
    top_idx = np.argsort(y_score)[::-1][:top_n]
    tdc = float(y_true[top_idx].sum() / max(int(y_true.sum()), 1))

    return ClassificationMetrics(
        roc_auc=roc,
        pr_auc=prauc,
        brier_score=brier,
        log_loss_val=ll,
        top_decile_capture=tdc,
        n_positive=int(y_true.sum()),
        n_total=len(y_true),
        split=split,
    )


def evaluate_severity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str = "",
    weights: np.ndarray | None = None,
) -> SeverityMetrics:
    """Compute severity regression metrics."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    w = weights[valid] if weights is not None else None
    err = yt - yp
    mae = float(np.average(np.abs(err), weights=w))
    rmse = float(np.sqrt(np.average(err**2, weights=w)))
    bias = float(np.average(err, weights=w))
    wa_mae = float(np.average(np.abs(err), weights=w)) if w is not None else None
    return SeverityMetrics(mae=mae, rmse=rmse, bias=bias, balance_weighted_mae=wa_mae, split=split)


def temporal_split(df, train_end: int, val_end: int, col: str = "origination_year"):
    """Split by origination vintage into train / val / test."""
    train = df.filter(df[col] <= train_end)
    val = df.filter((df[col] > train_end) & (df[col] <= val_end))
    test = df.filter(df[col] > val_end)
    return train, val, test
