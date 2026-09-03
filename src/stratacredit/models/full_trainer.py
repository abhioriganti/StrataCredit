"""Streaming full-universe logistic baselines over persisted DuckDB snapshots."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from stratacredit.db import get_connection
from stratacredit.models.dataset import build_model_snapshots
from stratacredit.models.evaluation import classification_metrics

console = Console()
FEATURES = [
    "current_upb", "original_upb", "original_interest_rate", "current_interest_rate",
    "fico", "original_ltv", "original_cltv", "dti", "loan_age", "remaining_term",
    "delinquency_months", "estimated_ltv", "modification_flag", "loan_purpose_code",
    "occupancy_code", "property_type_code", "reporting_month", "dq30_months_last_12m",
    "dq60_months_last_12m", "dq90_months_last_12m", "max_dq_last_12m", "ever_modified_last_12m",
]
BATCH_SIZE = 100_000
ARTIFACT_DIR = Path("artifacts/models")


def _batches(conn, where: str, target: str):
    # current_upb is already the first model feature and is reused as the
    # balance weight; selecting it twice creates duplicate Pandas columns.
    query = f"SELECT {', '.join(FEATURES)}, {target} FROM gold.model_snapshots WHERE {where}"
    for batch in conn.execute(query).fetch_record_batch(BATCH_SIZE):
        frame = batch.to_pandas()
        x = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        yield x, frame[target].to_numpy(dtype=np.int8), frame["current_upb"].to_numpy(dtype=np.float64)


def _fit_target(conn, target: str) -> tuple[StandardScaler, SGDClassifier, dict[str, float]]:
    train_where = "origination_year BETWEEN 2012 AND 2017"
    scaler = StandardScaler()
    negatives = positives = 0
    for x, y, _ in _batches(conn, train_where, target):
        scaler.partial_fit(x)
        positives += int(y.sum())
        negatives += int(len(y) - y.sum())
    if positives == 0:
        raise ValueError(f"No positive {target} labels in training data")
    event_weight = negatives / positives
    model = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=42)
    first = True
    for x, y, _ in _batches(conn, train_where, target):
        weights = np.where(y == 1, event_weight, 1.0)
        model.partial_fit(scaler.transform(x), y, classes=np.array([0, 1], dtype=np.int8), sample_weight=weights) if first else model.partial_fit(scaler.transform(x), y, sample_weight=weights)
        first = False
    return scaler, model, {"train_positive": float(positives), "train_negative": float(negatives)}


def run() -> dict[str, dict[str, float]]:
    """Materialize full snapshots and train streaming OOT logistic baselines."""
    build_model_snapshots()
    results: dict[str, dict[str, float]] = {}
    with get_connection("gold") as conn:
        metric_rows: list[dict[str, float | str]] = []
        for target in ("credit_event_12m", "prepayment_12m"):
            scaler, model, counts = _fit_target(conn, target)
            ys: list[np.ndarray] = []
            ps: list[np.ndarray] = []
            ws: list[np.ndarray] = []
            for x, y, weights in _batches(conn, "origination_year BETWEEN 2020 AND 2022", target):
                ys.append(y)
                ps.append(model.predict_proba(scaler.transform(x))[:, 1])
                ws.append(weights)
            results[target] = counts | classification_metrics(
                np.concatenate(ys).ravel(),
                np.concatenate(ps).ravel(),
                np.concatenate(ws).ravel(),
            )
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"target": target, "features": FEATURES, "scaler": scaler, "model": model},
                ARTIFACT_DIR / f"{target}_full_logistic.joblib",
            )
            metric_rows.append({"model_name": "full_logistic", "target": target} | results[target])
            console.print(f"[green]{target} OOT metrics: {results[target]}[/green]")
        metrics_frame = pd.DataFrame(metric_rows)
        conn.register("full_model_metrics_frame", metrics_frame)
        conn.execute("""
            CREATE OR REPLACE TABLE gold.gold_full_model_metrics AS
            SELECT * FROM full_model_metrics_frame
        """)
        conn.unregister("full_model_metrics_frame")
    return results


if __name__ == "__main__":
    run()
