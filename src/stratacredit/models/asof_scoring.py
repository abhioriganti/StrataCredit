"""As-of-date risk scoring for pool construction and stress analysis."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from stratacredit.db import get_connection
from stratacredit.models.full_trainer import FEATURES

ARTIFACT_DIR = Path("artifacts/models")


def score_as_of(cutoff_date: date) -> pl.DataFrame:
    """Return active-loan baseline/XGBoost scores using only data through cutoff."""
    with get_connection("silver") as conn:
        frame = conn.execute(f"""
            WITH history AS (
                SELECT
                    loan_id, reporting_period, current_upb, original_upb,
                    original_interest_rate, current_interest_rate, fico,
                    original_ltv, original_cltv, dti, loan_age, remaining_term,
                    mortgage_insurance_pct,
                    delinquency_months, estimated_ltv,
                    CAST(modification_flag AS INTEGER) AS modification_flag,
                    CASE loan_purpose WHEN 'P' THEN 0 WHEN 'C' THEN 1 WHEN 'R' THEN 2 ELSE 3 END AS loan_purpose_code,
                    CASE occupancy WHEN 'P' THEN 0 WHEN 'S' THEN 1 WHEN 'I' THEN 2 ELSE 3 END AS occupancy_code,
                    CASE property_type WHEN 'SF' THEN 0 WHEN 'CO' THEN 1 WHEN 'MH' THEN 2 WHEN 'PU' THEN 3 ELSE 4 END AS property_type_code,
                    MONTH(reporting_period) AS reporting_month,
                    SUM(CASE WHEN delinquency_months >= 1 THEN 1 ELSE 0 END) OVER loan_window AS dq30_months_last_12m,
                    SUM(CASE WHEN delinquency_months >= 2 THEN 1 ELSE 0 END) OVER loan_window AS dq60_months_last_12m,
                    SUM(CASE WHEN delinquency_months >= 3 THEN 1 ELSE 0 END) OVER loan_window AS dq90_months_last_12m,
                    MAX(delinquency_months) OVER loan_window AS max_dq_last_12m,
                    MAX(CAST(modification_flag AS INTEGER)) OVER loan_window AS ever_modified_last_12m
                FROM silver.loan_month_panel
                WHERE reporting_period <= CAST('{cutoff_date}' AS DATE)
                WINDOW loan_window AS (PARTITION BY loan_id ORDER BY reporting_period ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)
            )
            SELECT * FROM history WHERE reporting_period = CAST('{cutoff_date}' AS DATE)
        """).pl()
    if frame.is_empty():
        return pl.DataFrame({"loan_id": [], "credit_event_score": [], "prepayment_score": []})
    pandas_frame = frame.to_pandas()
    x = pandas_frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    credit = joblib.load(ARTIFACT_DIR / "credit_event_12m_full_logistic.joblib")
    prepay = joblib.load(ARTIFACT_DIR / "prepayment_12m_full_logistic.joblib")
    result = pl.DataFrame({
        "loan_id": pandas_frame["loan_id"],
        "credit_event_score": credit["model"].predict_proba(credit["scaler"].transform(x))[:, 1],
        "prepayment_score": prepay["model"].predict_proba(prepay["scaler"].transform(x))[:, 1],
    })
    for target, alias in (("credit_event_12m", "credit_event_xgb_score"), ("prepayment_12m", "prepayment_xgb_score")):
        path = ARTIFACT_DIR / f"{target}_xgboost.joblib"
        if path.exists():
            result = result.with_columns(pl.Series(alias, joblib.load(path)["model"].predict_proba(x)[:, 1]))
    severity_path = ARTIFACT_DIR / "loss_severity_ridge.joblib"
    if severity_path.exists():
        severity = joblib.load(severity_path)
        result = result.with_columns(
            pl.Series(
                "loss_severity_score",
                np.clip(severity["model"].predict(pandas_frame[severity["features"]]), 0.0, 1.5),
            )
        )
    return result
