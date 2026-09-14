"""Real-data XGBoost AFT model for time to credit event with censoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from stratacredit.db import get_connection

FEATURES = [
    "fico",
    "original_ltv",
    "original_cltv",
    "dti",
    "original_upb",
    "original_interest_rate",
    "mortgage_insurance_pct",
]


def run() -> dict[str, float]:
    with get_connection("silver") as conn:
        df = conn.execute("""
            WITH events AS (SELECT loan_id, MIN(loan_age) AS event_age FROM silver.loan_month_panel WHERE is_credit_event GROUP BY loan_id),
            observed AS (SELECT loan_id, MAX(loan_age) AS last_age FROM silver.loan_month_panel GROUP BY loan_id),
            orig AS (SELECT DISTINCT loan_id, origination_year, fico, original_ltv, original_cltv, dti, original_upb, original_interest_rate, mortgage_insurance_pct FROM silver.loan_month_panel)
            SELECT orig.*, COALESCE(events.event_age, observed.last_age) AS time_lower,
                   CASE WHEN events.event_age IS NULL THEN 1e9 ELSE events.event_age END AS time_upper,
                   CASE WHEN events.event_age IS NULL THEN 0 ELSE 1 END AS event_observed
            FROM orig JOIN observed USING (loan_id) LEFT JOIN events USING (loan_id)
        """).df()
    train, test = df[df.origination_year <= 2019], df[df.origination_year >= 2020]
    x_train = train[FEATURES].fillna(train[FEATURES].median()).to_numpy(dtype=np.float32)
    x_test = test[FEATURES].fillna(train[FEATURES].median()).to_numpy(dtype=np.float32)
    dtrain, dtest = xgb.DMatrix(x_train), xgb.DMatrix(x_test)
    dtrain.set_float_info("label_lower_bound", train.time_lower.to_numpy(dtype=np.float32))
    dtrain.set_float_info("label_upper_bound", train.time_upper.to_numpy(dtype=np.float32))
    model = xgb.train(
        {
            "objective": "survival:aft",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "tree_method": "hist",
            "seed": 42,
        },
        dtrain,
        num_boost_round=150,
    )
    pred = model.predict(dtest)
    metrics = {
        "model_name": "xgboost_aft_credit",
        "train_loans": float(len(train)),
        "test_loans": float(len(test)),
        "train_events": float(train.event_observed.sum()),
        "test_events": float(test.event_observed.sum()),
        "median_predicted_time_event": float(np.median(pred[test.event_observed.to_numpy() == 1]))
        if test.event_observed.sum()
        else float("nan"),
        "median_predicted_time_censored": float(
            np.median(pred[test.event_observed.to_numpy() == 0])
        ),
    }
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"features": FEATURES, "model": model}, "artifacts/models/survival_credit_aft.joblib"
    )
    with get_connection("gold") as conn:
        import pandas as pd

        conn.register("survival_metrics", pd.DataFrame([metrics]))
        conn.execute(
            "CREATE OR REPLACE TABLE gold.gold_survival_credit_metrics AS SELECT * FROM survival_metrics"
        )
        conn.unregister("survival_metrics")
    return metrics


if __name__ == "__main__":
    print(run())
