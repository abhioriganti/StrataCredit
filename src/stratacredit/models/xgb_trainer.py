"""Temporal XGBoost challengers using deterministic, event-preserving samples."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from rich.console import Console

from stratacredit.db import get_connection
from stratacredit.models.evaluation import classification_metrics
from stratacredit.models.full_trainer import FEATURES

console = Console()
ARTIFACT_DIR = Path("artifacts/models")


def _load_split(conn, years: str, target: str, sample_pct: int | None) -> pd.DataFrame:
    sample_filter = ""
    if sample_pct is not None:
        # All events are retained; non-events are deterministically sampled by
        # loan/date hash so repeated runs produce the same training set.
        sample_filter = (
            f" AND ({target} = 1 OR MOD(ABS(HASH(loan_id, scoring_date)), 100) < {sample_pct})"
        )
    query = f"""
        SELECT {', '.join(FEATURES)}, {target}
        FROM gold.model_snapshots
        WHERE origination_year {years}{sample_filter}
    """
    return conn.execute(query).fetchdf()


def _xy(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return x, frame[target].to_numpy(dtype=np.int8), frame["current_upb"].to_numpy(dtype=np.float64)


def _fit_target(conn, target: str, sample_pct: int) -> dict[str, float | str]:
    train = _load_split(conn, "BETWEEN 2012 AND 2017", target, sample_pct)
    validation = _load_split(conn, "BETWEEN 2018 AND 2019", target, sample_pct)
    test = _load_split(conn, "BETWEEN 2020 AND 2022", target, None)
    x_train, y_train, _ = _xy(train, target)
    x_val, y_val, _ = _xy(validation, target)
    x_test, y_test, weights = _xy(test, target)
    if y_train.sum() == 0 or y_val.sum() == 0:
        raise ValueError(f"Insufficient positive {target} observations for temporal XGBoost fit")
    scale_pos_weight = float((y_train == 0).sum()) / float(y_train.sum())
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.85,
        min_child_weight=25,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=35,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, probability, weights)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"target": target, "features": FEATURES, "model": model, "sample_pct": sample_pct},
        ARTIFACT_DIR / f"{target}_xgboost.joblib",
    )
    importance = pd.DataFrame(
        {"target": target, "feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    conn.register("xgb_importance", importance)
    conn.execute("CREATE OR REPLACE TABLE gold.gold_xgboost_feature_importance AS SELECT * FROM xgb_importance") if target == "credit_event_12m" else conn.execute("INSERT INTO gold.gold_xgboost_feature_importance SELECT * FROM xgb_importance")
    conn.unregister("xgb_importance")
    return {
        "model_name": "xgboost_temporal_challenger",
        "target": target,
        "train_rows": float(len(train)),
        "validation_rows": float(len(validation)),
        "test_rows": float(len(test)),
        "best_iteration": float(model.best_iteration),
    } | metrics


def run() -> dict[str, dict[str, float | str]]:
    """Fit XGBoost challengers and persist their real OOT comparison metrics."""
    results: dict[str, dict[str, float | str]] = {}
    with get_connection("gold") as conn:
        for target, sample_pct in (("credit_event_12m", 10), ("prepayment_12m", 1)):
            results[target] = _fit_target(conn, target, sample_pct)
            console.print(f"[green]{target} XGBoost OOT: {results[target]}[/green]")
        frame = pd.DataFrame(results.values())
        conn.register("xgb_metrics", frame)
        conn.execute("CREATE OR REPLACE TABLE gold.gold_xgboost_model_metrics AS SELECT * FROM xgb_metrics")
        conn.unregister("xgb_metrics")
    return results


if __name__ == "__main__":
    run()
