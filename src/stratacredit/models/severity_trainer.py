"""Temporal conditional loss-severity baseline on real terminal credit events."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stratacredit.db import get_connection

FEATURES = [
    "fico",
    "original_ltv",
    "original_cltv",
    "dti",
    "loan_age",
    "original_upb",
    "mortgage_insurance_pct",
]
ARTIFACT = Path("artifacts/models/loss_severity_ridge.joblib")


def run() -> dict[str, float]:
    with get_connection("silver") as conn:
        frame = conn.execute("""
            SELECT fico, original_ltv, original_cltv, dti, loan_age, original_upb,
                   mortgage_insurance_pct, zero_balance_removal_upb,
                   actual_loss / NULLIF(zero_balance_removal_upb, 0) AS loss_severity,
                   origination_year
            FROM silver.loan_month_panel
            WHERE is_credit_event
              AND actual_loss IS NOT NULL
              AND zero_balance_removal_upb > 0
              AND actual_loss / zero_balance_removal_upb BETWEEN 0 AND 1.5
        """).df()
    train, test = frame[frame.origination_year <= 2019], frame[frame.origination_year >= 2020]
    if len(train) < 10 or len(test) < 1:
        raise ValueError("Insufficient realized credit events for temporal severity validation")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train[FEATURES], train.loss_severity)
    prediction = np.clip(model.predict(test[FEATURES]), 0, 1.5)
    y, weights = test.loss_severity.to_numpy(), test.zero_balance_removal_upb.to_numpy()
    metrics = {
        "model_name": "ridge_conditional_severity",
        "events_train": float(len(train)),
        "events_test": float(len(test)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "bias": float(np.mean(prediction - y)),
        "balance_weighted_mae": float(np.average(np.abs(y - prediction), weights=weights)),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": FEATURES, "model": model}, ARTIFACT)
    with get_connection("gold") as conn:
        conn.register("severity_metrics", pd.DataFrame([metrics]))
        conn.execute(
            "CREATE OR REPLACE TABLE gold.gold_loss_severity_model_metrics AS SELECT * FROM severity_metrics"
        )
        conn.unregister("severity_metrics")
    return metrics


if __name__ == "__main__":
    print(run())
