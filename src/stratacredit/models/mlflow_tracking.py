"""Register persisted models, OOT metrics, and diagnostics in local MLflow."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from stratacredit.config import models_config
from stratacredit.db import get_connection

ARTIFACT_DIR = Path("artifacts/models")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def register_runs() -> None:
    """Log existing baseline and challenger artifacts without retraining."""
    mlflow.set_tracking_uri(models_config["mlflow"]["tracking_uri"])
    with get_connection("gold") as conn:
        baseline = conn.execute("SELECT * FROM gold.gold_full_model_metrics").df()
        challenger = conn.execute("SELECT * FROM gold.gold_xgboost_model_metrics").df()
        scores = conn.execute("SELECT * FROM gold.gold_model_scores").df()
    for _, row in pd.concat([baseline, challenger], ignore_index=True).iterrows():
        target, model_name = row["target"], row["model_name"]
        experiment = mlflow.set_experiment(target)
        score_col = "credit_event_score" if target == "credit_event_12m" else "prepayment_score"
        if model_name.startswith("xgboost"):
            score_col = "credit_event_xgb_score" if target == "credit_event_12m" else "prepayment_xgb_score"
            artifact = ARTIFACT_DIR / f"{target}_xgboost.joblib"
        else:
            artifact = ARTIFACT_DIR / f"{target}_full_logistic.joblib"
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name):
            mlflow.set_tags({"model_type": model_name, "train_vintage_range": "2012-2017", "validation_vintage_range": "2018-2019", "test_vintage_range": "2020-2022", "target_definition": f"{target} in t+1 through t+12", "feature_set_version": "snapshot_v1", "git_commit_sha": _git_sha()})
            mlflow.log_params({"artifact": artifact.name, "feature_count": 22})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.number)) and pd.notna(v)})
            sample = scores[[target, score_col]].dropna()
            with tempfile.TemporaryDirectory() as directory:
                directory_path = Path(directory)
                actual, predicted = sample[target].to_numpy(), sample[score_col].to_numpy()
                fraction, mean_prediction = calibration_curve(actual, predicted, n_bins=10, strategy="quantile")
                fig, ax = plt.subplots(); ax.plot(mean_prediction, fraction, marker="o"); ax.plot([0, 1], [0, 1], "--"); ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate", title=f"{target} calibration"); fig.savefig(directory_path / "calibration_plot.png", bbox_inches="tight"); plt.close(fig)
                fig, ax = plt.subplots(); RocCurveDisplay.from_predictions(actual, predicted, ax=ax); fig.savefig(directory_path / "roc_curve.png", bbox_inches="tight"); plt.close(fig)
                fig, ax = plt.subplots(); PrecisionRecallDisplay.from_predictions(actual, predicted, ax=ax); fig.savefig(directory_path / "pr_curve.png", bbox_inches="tight"); plt.close(fig)
                scores.groupby("origination_year").agg(realized=(target, "mean"), predicted=(score_col, "mean")).to_csv(directory_path / "cohort_performance.csv")
                shutil.copy2(artifact, directory_path / artifact.name)
                mlflow.log_artifacts(str(directory_path))


if __name__ == "__main__":
    register_runs()
