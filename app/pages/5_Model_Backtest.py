"""Page 5: Model & Backtest — OOT metrics, ROC/PR, feature importance."""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model & Backtest", layout="wide")
st.title("🤖 Model & Backtest")
st.caption("Out-of-time validation results and model diagnostics.")


@st.cache_data(ttl=120)
def load_gold_table(table: str) -> pd.DataFrame:
    """Load a persisted model result table, returning an empty frame if absent."""
    try:
        from stratacredit.db import get_connection

        with get_connection("gold") as conn:
            return conn.execute(f"SELECT * FROM gold.{table}").df()
    except Exception:
        return pd.DataFrame()


st.subheader("Persisted Full-Universe Results")
baseline_metrics = load_gold_table("gold_full_model_metrics")
xgb_metrics = load_gold_table("gold_xgboost_model_metrics")
metric_frames = [frame for frame in (baseline_metrics, xgb_metrics) if not frame.empty]
if metric_frames:
    metrics = pd.concat(metric_frames, ignore_index=True)
    display_columns = [
        column
        for column in [
            "model_name",
            "target",
            "train_rows",
            "validation_rows",
            "test_rows",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
        ]
        if column in metrics.columns
    ]
    st.dataframe(metrics[display_columns], use_container_width=True, hide_index=True)
    if {"model_name", "target", "roc_auc"}.issubset(metrics.columns):
        st.plotly_chart(
            px.bar(
                metrics,
                x="target",
                y="roc_auc",
                color="model_name",
                barmode="group",
                title="OOT ROC-AUC by model",
            ),
            use_container_width=True,
        )
else:
    st.info("No persisted model metrics found. Run `make train`, `make xgb`, then `make score`.")

backtest_target = st.selectbox(
    "Backtest target", ["credit_event", "prepayment"], key="persisted_backtest_target"
)
backtest = load_gold_table(f"gold_{backtest_target}_12m_backtest")
if not backtest.empty:
    aggregate = (
        backtest.groupby("scoring_date", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "realized_rate": (group["realized_rate"] * group["current_upb"]).sum()
                    / group["current_upb"].sum(),
                    "predicted_rate": (group["predicted_rate"] * group["current_upb"]).sum()
                    / group["current_upb"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    curve = aggregate.melt("scoring_date", var_name="series", value_name="rate")
    st.plotly_chart(
        px.line(
            curve,
            x="scoring_date",
            y="rate",
            color="series",
            markers=True,
            title="Balance-weighted predicted vs realized 12-month rate",
        ),
        use_container_width=True,
    )

importance = load_gold_table("gold_xgboost_feature_importance")
if not importance.empty:
    st.plotly_chart(
        px.bar(
            importance.sort_values("importance", ascending=False).head(15),
            x="importance",
            y="feature",
            color="target",
            orientation="h",
            barmode="group",
            title="Top XGBoost feature importance",
        ),
        use_container_width=True,
    )

st.subheader("Leakage Review")
leakage_audit = load_gold_table("gold_model_leakage_audit")
if not leakage_audit.empty:
    st.dataframe(leakage_audit, use_container_width=True, hide_index=True)
    st.caption(
        "Credit-event XGBoost is dominated by current and trailing delinquency features. "
        "It is an active-universe servicing-risk model; evaluate clean-pool performance separately."
    )

st.divider()

# ── MLflow runs ───────────────────────────────────────────────────────────────
st.subheader("MLflow Experiment Runs")


@st.cache_data(ttl=120)
def load_mlflow_runs():
    try:
        import mlflow

        from stratacredit.config import models_config

        mlflow.set_tracking_uri(models_config["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        rows = []
        for exp_name in ["credit_event_12m", "prepayment_12m", "loss_severity"]:
            try:
                exp = client.get_experiment_by_name(exp_name)
                if exp is None:
                    continue
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=10,
                )
                for run in runs:
                    m = run.data.metrics
                    rows.append(
                        {
                            "Experiment": exp_name,
                            "Run": run.data.tags.get("model_type", run.info.run_name),
                            "Train AUC": round(m.get("train_roc_auc", float("nan")), 4),
                            "Val AUC": round(
                                m.get("val_roc_auc", m.get("roc_auc", float("nan"))), 4
                            ),
                            "Test AUC": round(
                                m.get("test_roc_auc", m.get("roc_auc", float("nan"))), 4
                            ),
                            "Val PR-AUC": round(
                                m.get("val_pr_auc", m.get("pr_auc", float("nan"))), 4
                            ),
                            "Val Brier": round(
                                m.get("val_brier_score", m.get("brier_score", float("nan"))), 4
                            ),
                            "Run ID": run.info.run_id[:8],
                        }
                    )
            except Exception:
                pass
        return rows
    except Exception:
        return []


runs = load_mlflow_runs()
if runs:
    import pandas as pd

    st.dataframe(pd.DataFrame(runs), use_container_width=True)
else:
    st.info("No MLflow runs found. Run `make train` to train models.")

st.divider()

# ── Quick backtest on fixture data ────────────────────────────────────────────
st.subheader("Quick Backtest (Fixture Data)")
st.caption("Trains on fixture data and evaluates classification metrics.")

if st.button("▶ Run Quick Backtest"):
    with st.spinner("Loading panel and training..."):
        try:
            from stratacredit.db import get_connection

            with get_connection("silver") as conn:
                panel = conn.execute("SELECT * FROM silver.loan_month_panel").pl()

            from stratacredit.features.snapshot import build_training_dataset

            train = build_training_dataset(panel, 2018, 2020)
            val = build_training_dataset(panel, 2021, 2021)
            test = build_training_dataset(panel, 2022, 2022)

            if len(train) == 0:
                st.warning("No training data. Ensure Silver tables are built.")
            else:
                results = []
                for target in ("credit_event_12m", "prepayment_12m"):
                    from stratacredit.models.logistic import score_logistic, train_logistic

                    pipe, feat_cols, tr_m = train_logistic(train, target)
                    if len(val) > 0:
                        _, val_m = score_logistic(pipe, val, feat_cols, target, "val")
                        if val_m:
                            results.append(
                                {
                                    "Target": target,
                                    "Split": "Validation",
                                    "ROC-AUC": round(val_m.roc_auc, 4),
                                    "PR-AUC": round(val_m.pr_auc, 4),
                                    "Brier": round(val_m.brier_score, 4),
                                    "N+": val_m.n_positive,
                                }
                            )
                    if len(test) > 0:
                        _, te_m = score_logistic(pipe, test, feat_cols, target, "test")
                        if te_m:
                            results.append(
                                {
                                    "Target": target,
                                    "Split": "Test",
                                    "ROC-AUC": round(te_m.roc_auc, 4),
                                    "PR-AUC": round(te_m.pr_auc, 4),
                                    "Brier": round(te_m.brier_score, 4),
                                    "N+": te_m.n_positive,
                                }
                            )
                import pandas as pd

                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.warning("No labeled validation/test data available.")
        except Exception as e:
            st.error(f"Backtest error: {e}")
