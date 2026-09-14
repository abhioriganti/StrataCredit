"""Model evaluation functions used by training and backtests."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Return stable classification metrics; undefined metrics are NaN."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    result = {
        "brier_score": float(brier_score_loss(y, p, sample_weight=weights)),
        # Explicit labels keeps log loss defined for sparse OOT cohorts with
        # only non-events (or only events).
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weights)),
    }
    if len(np.unique(y)) < 2:
        return result | {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "top_decile_capture_rate": float("nan"),
        }
    result["roc_auc"] = float(roc_auc_score(y, p, sample_weight=weights))
    result["pr_auc"] = float(average_precision_score(y, p, sample_weight=weights))
    cutoff = np.quantile(p, 0.9)
    positives = y.sum()
    result["top_decile_capture_rate"] = (
        float(y[p >= cutoff].sum() / positives) if positives else float("nan")
    )
    return result


def portfolio_backtest(
    df: pl.DataFrame, target: str, score: str, dimensions: list[str]
) -> pl.DataFrame:
    """Compare mean predicted and realized event rates for valid cohorts."""
    dims = [c for c in dimensions if c in df.columns]
    if not dims:
        dims = ["scoring_date"] if "scoring_date" in df.columns else []
    return (
        df.group_by(dims)
        .agg(
            pl.len().alias("observations"),
            pl.col(target).mean().alias("realized_rate"),
            pl.col(score).mean().alias("predicted_rate"),
            (pl.col(score).mean() - pl.col(target).mean()).alias("prediction_bias"),
        )
        .sort(dims)
    )
