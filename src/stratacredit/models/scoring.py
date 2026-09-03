"""Persist full-universe out-of-time model scores and portfolio backtests."""

from __future__ import annotations

from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

from stratacredit.models.full_trainer import BATCH_SIZE, FEATURES
from stratacredit.db import get_connection

console = Console()
ARTIFACT_DIR = Path("artifacts/models")
HOLDOUT_FILTER = "origination_year BETWEEN 2020 AND 2022"


def _score_batches(conn):
    credit = joblib.load(ARTIFACT_DIR / "credit_event_12m_full_logistic.joblib")
    prepay = joblib.load(ARTIFACT_DIR / "prepayment_12m_full_logistic.joblib")
    credit_xgb_path = ARTIFACT_DIR / "credit_event_12m_xgboost.joblib"
    prepay_xgb_path = ARTIFACT_DIR / "prepayment_12m_xgboost.joblib"
    credit_xgb = joblib.load(credit_xgb_path) if credit_xgb_path.exists() else None
    prepay_xgb = joblib.load(prepay_xgb_path) if prepay_xgb_path.exists() else None
    query = f"""
        SELECT loan_id, scoring_date, origination_year, {', '.join(FEATURES)},
               credit_event_12m, prepayment_12m
        FROM gold.model_snapshots
        WHERE {HOLDOUT_FILTER}
    """
    for batch in conn.execute(query).fetch_record_batch(BATCH_SIZE):
        frame = batch.to_pandas()
        x = (
            frame[FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        scored = {
                "loan_id": frame["loan_id"],
                "scoring_date": frame["scoring_date"],
                "origination_year": frame["origination_year"],
                "current_upb": frame["current_upb"],
                "fico": frame["fico"],
                "original_ltv": frame["original_ltv"],
                "credit_event_12m": frame["credit_event_12m"],
                "prepayment_12m": frame["prepayment_12m"],
                "credit_event_score": credit["model"].predict_proba(
                    credit["scaler"].transform(x)
                )[:, 1],
                "prepayment_score": prepay["model"].predict_proba(
                    prepay["scaler"].transform(x)
                )[:, 1],
        }
        if credit_xgb is not None:
            scored["credit_event_xgb_score"] = credit_xgb["model"].predict_proba(x)[:, 1]
        if prepay_xgb is not None:
            scored["prepayment_xgb_score"] = prepay_xgb["model"].predict_proba(x)[:, 1]
        yield pd.DataFrame(scored)


def persist_scores() -> None:
    """Score all OOT snapshots in batches and create portfolio backtest tables."""
    missing = [
        path for path in (
            ARTIFACT_DIR / "credit_event_12m_full_logistic.joblib",
            ARTIFACT_DIR / "prepayment_12m_full_logistic.joblib",
        ) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Run `make train` before creating scores.")

    with get_connection("gold") as conn:
        total = 0
        # Do not issue writes on ``conn`` while its Arrow reader is active:
        # DuckDB invalidates that reader and silently limits output to the first
        # batch.  Stage bounded batches as Parquet, then import after reading.
        with tempfile.TemporaryDirectory(prefix="model_scores_", dir="artifacts") as staging:
            staged_paths: list[str] = []
            for batch_number, score_frame in enumerate(_score_batches(conn)):
                staged_path = Path(staging) / f"scores_{batch_number:05d}.parquet"
                score_frame.to_parquet(staged_path, index=False)
                staged_paths.append(staged_path.as_posix())
                total += len(score_frame)
            if not staged_paths:
                raise ValueError("No out-of-time snapshots available to score")
            parquet_glob = (Path(staging) / "*.parquet").as_posix()
            conn.execute(
                f"CREATE OR REPLACE TABLE gold.gold_model_scores AS "
                f"SELECT * FROM read_parquet('{parquet_glob}')"
            )
        conn.execute("""
            CREATE OR REPLACE TABLE gold.gold_credit_event_12m_backtest AS
            SELECT
                scoring_date,
                origination_year,
                COUNT(*) AS observations,
                SUM(current_upb) AS current_upb,
                AVG(credit_event_12m) AS realized_rate,
                AVG(credit_event_score) AS predicted_rate,
                AVG(credit_event_score) - AVG(credit_event_12m) AS prediction_bias,
                SUM(current_upb * credit_event_12m) / NULLIF(SUM(current_upb), 0) AS balance_weighted_realized_rate,
                SUM(current_upb * credit_event_score) / NULLIF(SUM(current_upb), 0) AS balance_weighted_predicted_rate
            FROM gold.gold_model_scores
            GROUP BY scoring_date, origination_year
        """)
        conn.execute("""
            CREATE OR REPLACE TABLE gold.gold_prepayment_12m_backtest AS
            SELECT
                scoring_date,
                origination_year,
                COUNT(*) AS observations,
                SUM(current_upb) AS current_upb,
                AVG(prepayment_12m) AS realized_rate,
                AVG(prepayment_score) AS predicted_rate,
                AVG(prepayment_score) - AVG(prepayment_12m) AS prediction_bias,
                SUM(current_upb * prepayment_12m) / NULLIF(SUM(current_upb), 0) AS balance_weighted_realized_rate,
                SUM(current_upb * prepayment_score) / NULLIF(SUM(current_upb), 0) AS balance_weighted_predicted_rate
            FROM gold.gold_model_scores
            GROUP BY scoring_date, origination_year
        """)
    console.print(f"[green]Persisted {total:,} OOT model scores and backtests.[/green]")


if __name__ == "__main__":
    persist_scores()
