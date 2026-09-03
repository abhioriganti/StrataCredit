# Model Card

## Models Overview

StrataCredit trains four model types, all evaluated with out-of-time
temporal validation. No random train/test splits are used as the
primary evaluation.

---

## Model 1: Logistic Regression (Baseline)

**Targets**: `credit_event_12m`, `prepayment_12m`

**Purpose**: Interpretable baseline. Establishes a performance floor
and provides coefficient-based feature importance.

**Data**: Leakage-safe model snapshots from the Silver panel.

**Features**: Origination characteristics + delinquency history as of
the scoring date. No post-event fields.

**Temporal Split**:
- Train: 2012–2017 origination vintages
- Validation: 2018–2019
- Test: 2020–2022

**Preprocessing**: Median imputation → StandardScaler → LogisticRegression.

**Class imbalance**: `class_weight="balanced"`.

**Metrics**: ROC-AUC, PR-AUC, Brier score, log loss, top-decile capture.

**Limitations**: Assumes linear decision boundary. May underfit nonlinear
interactions between FICO, LTV, and loan age.

---

## Model 2: XGBoost

**Targets**: `credit_event_12m`, `prepayment_12m`

**Purpose**: Captures nonlinear interactions. Primary production model
if it outperforms logistic baseline.

**Features**: Same as logistic + macro context (vintage year, reporting year).

**Hyperparameters**: See `configs/models.yaml` [xgboost] section.

**Early stopping**: Validation PR-AUC with 50 rounds patience.

**Class imbalance**: `scale_pos_weight` set to class ratio.

**Limitations**: Less interpretable than logistic. SHAP values available
but not automatically computed in V1.

### Current Full-Universe Run and Leakage Review

The current run materialized 10,084,128 quarterly snapshots and used the
following originations-vintage split: training 2012-2017, validation
2018-2019, and untouched test 2020-2022. The Gold tables
`gold_model_leakage_audit` and `gold_credit_event_cohort_audit` record the
checks and cohort outcomes for each rebuild.

The audit passed its hard checks: no forbidden termination, loss, proceeds,
recovery, expense, or target fields appeared in the feature list, and no
terminal row was eligible for scoring. The 12-month target is constructed
only from rows after the scoring date.

On the active-universe credit-event test, XGBoost achieved a high ROC-AUC.
This should not be interpreted as a broad performing-loan result: current
delinquency and trailing delinquency counts account for most model
importance. That is legitimate information available at the scoring date,
but it means the model primarily ranks servicing distress. A pool with a
current-loan eligibility screen has very few observed credit events in the
sample, so its clean-pool discrimination must be reported separately rather
than inferred from the active-universe metric.

---

## Model 3: XGBoost AFT Survival Model

**Targets**: Time to voluntary prepayment; time to credit event.

**Purpose**: Estimates expected time-to-event rather than binary 12-month flag.
Handles right-censoring correctly.

**Censoring**: Loans not yet terminated are right-censored at last observed month.

**Distribution**: Normal AFT (configurable in `configs/models.yaml`).

**Limitations**: AFT assumes a parametric time distribution. Complex
time-varying covariates are approximated by last-observed-value imputation.

---

## Model 4: Loss Severity

**Target**: `loss_severity` = actual_loss / zero_balance_removal_upb

**Purpose**: Predicts realized severity conditional on a credit event.

**Sample**: Only observations with qualifying terminal credit events
and non-null actual_loss and zero_balance_removal_upb.

**Forbidden features**: All post-event recovery fields (net_sale_proceeds,
mi_recoveries, actual_loss, etc.) are excluded.

**Severity bounds**: Predictions clipped to [0.0, 1.5].

**Metrics**: MAE, RMSE, bias, balance-weighted MAE.

**Limitations**: Sample size is limited by historical credit event frequency.
Small event counts in recent vintages may reduce reliability.

---

## General Limitations

- All models are trained on Freddie Mac conforming conventional mortgages.
  They are not validated on non-QM, jumbo, FHA, VA, or other loan types.
- Historical performance from 2012–2022 includes a period of sustained
  house price appreciation and low rates. Model behavior in a sustained
  price-decline environment may differ from historical patterns.
- These models are for analytical purposes only. They are NOT credit
  ratings and do NOT constitute investment recommendations.
