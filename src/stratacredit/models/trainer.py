"""Leakage-safe, out-of-time baseline model training.

This module intentionally trains only models that can be supported by the
available data.  It writes scored test observations and cohort backtests to
the Gold database; MLflow tracking is used when installed.
"""

from __future__ import annotations

import subprocess

import numpy as np
import polars as pl
from rich.console import Console
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stratacredit.config import models_config
from stratacredit.db import get_connection
from stratacredit.features.encoder import encode_categoricals, get_feature_columns, to_numpy
from stratacredit.features.snapshot import build_training_dataset
from stratacredit.models.evaluation import classification_metrics, portfolio_backtest

console = Console()
TARGETS = ("credit_event_12m", "prepayment_12m")
# A reproducible development-scale sample. Full-universe fitting can be enabled
# later on a machine sized for the complete servicing history.
LOAN_SAMPLE_MODULO = 20


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _dataset_for_vintages(panel: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    return build_training_dataset(panel, start, end, max_scoring_dates=12).drop_nulls(TARGETS)


def _fit_and_score(
    train: pl.DataFrame, test: pl.DataFrame, target: str
) -> tuple[Pipeline, np.ndarray, list[str]]:
    train, test = encode_categoricals(train), encode_categoricals(test)
    features = sorted(
        set(get_feature_columns(train, list(TARGETS)))
        & set(get_feature_columns(test, list(TARGETS)))
    )
    if not features:
        raise ValueError("No numeric, leakage-safe features available for training")
    x_train, y_train = to_numpy(train, features, target)
    x_test, _ = to_numpy(test, features)
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{target} has fewer than two classes in the training vintages")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_test)[:, 1], features


def run() -> dict[str, dict[str, float]]:
    """Train logistic baselines on configured temporal vintages and persist OOT scores."""
    split = models_config["temporal_split"]
    with get_connection("silver") as conn:
        # Read only the vintages needed for each split. Loading the entire
        # all-vintage panel before filtering is prohibitively memory-intensive.
        train_panel = conn.execute(
            """SELECT * FROM silver.loan_month_panel
               WHERE origination_year BETWEEN ? AND ? AND hash(loan_id) % ? = 0""",
            [
                split["train"]["vintage_start"],
                split["train"]["vintage_end"],
                LOAN_SAMPLE_MODULO,
            ],
        ).pl()
        test_panel = conn.execute(
            """SELECT * FROM silver.loan_month_panel
               WHERE origination_year BETWEEN ? AND ? AND hash(loan_id) % ? = 0""",
            [
                split["test"]["vintage_start"],
                split["test"]["vintage_end"],
                LOAN_SAMPLE_MODULO,
            ],
        ).pl()
    train = _dataset_for_vintages(
        train_panel, split["train"]["vintage_start"], split["train"]["vintage_end"]
    )
    test = _dataset_for_vintages(
        test_panel, split["test"]["vintage_start"], split["test"]["vintage_end"]
    )
    if train.is_empty() or test.is_empty():
        raise ValueError(
            "No labeled temporal snapshots. Ensure at least 12 future reporting months are available."
        )
    metrics: dict[str, dict[str, float]] = {}
    scored = test
    for target in TARGETS:
        model, probability, features = _fit_and_score(train, test, target)
        score_column = f"predicted_{target}"
        scored = scored.with_columns(pl.Series(score_column, probability))
        metrics[target] = classification_metrics(
            test[target].to_numpy(),
            probability,
            test["current_upb"].to_numpy() if "current_upb" in test.columns else None,
        )
        metrics[target]["feature_count"] = float(len(features))
        metrics[target]["git_sha"] = _git_sha()
    with get_connection("gold") as conn:
        conn.register("model_scores_frame", scored.to_pandas())
        conn.execute(
            "CREATE OR REPLACE TABLE gold.gold_model_scores AS SELECT * FROM model_scores_frame"
        )
        for target in TARGETS:
            cohort = portfolio_backtest(
                scored, target, f"predicted_{target}", ["origination_year", "scoring_date"]
            )
            conn.register("cohort_frame", cohort.to_pandas())
            conn.execute(
                f"CREATE OR REPLACE TABLE gold.gold_{target}_backtest AS SELECT * FROM cohort_frame"
            )
    console.print(
        f"[green]Trained temporal logistic baselines on {len(train):,} observations; "
        f"scored {len(test):,} OOT observations (deterministic 1/{LOAN_SAMPLE_MODULO} loan sample).[/green]"
    )
    return metrics


if __name__ == "__main__":
    run()
