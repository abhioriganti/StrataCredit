# StrataCredit

StrataCredit is a practical loan-level structured-credit analytics workbench built with the Freddie Mac Single-Family Loan-Level Dataset. It takes raw origination and servicing files, validates and standardizes them, helps an analyst understand the collateral and its historical performance, and supports pool selection, stress testing, and exportable deal analysis.

## What you can do with StrataCredit

- Raw-file metadata, checksum capture, typed Bronze Parquet, and DuckDB Silver/Gold layers.
- Canonical loan-month panel, validation rules, quarantine generation, collateral metrics, stratifications, replines, CPR, delinquency transitions, credit-event and severity analysis.
- Configuration-driven greedy pool selection with constraint reconciliation, candidate-versus-selected comparison, scenarios, CSV exports, and HTML memo.
- Leakage-safe model snapshots and temporal logistic-regression baselines with OOT score and cohort-backtest persistence.
- A six-page Streamlit workbench and synthetic fixture-based test suite.

## Quick start with GNU Make

```powershell
cd C:\Users\abhis\projects\StrataCredit
py -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
make fixture
make pipeline
make train
make xgb
make score
make mlflow
make leakage-audit
make pool
make scenarios
make report
make app
```

`make fixture` produces a deterministic synthetic tape for tests. For real processing, register for the Freddie Mac dataset, place the unmodified `sample_orig_*.txt` and `sample_svcg_*.txt` files under `data/raw/<vintage>/`, then run `make ingest`. Raw files, derived data, models, and outputs are ignored by Git. If you are on Windows without GNU Make, use the PowerShell commands below instead.

## Model workflow

`make train` fits full-universe streaming logistic baselines. `make xgb` fits the
temporal XGBoost challengers, `make score` persists out-of-time loan-level
scores and backtests, and `make mlflow` registers models, diagnostics, and
calibration artifacts in the local `mlflow.db` tracking store. `make
leakage-audit` records feature and temporal checks in Gold. The survival and
severity model runners are available under `stratacredit.models` and their
metrics are persisted to Gold after training.

## A note on scope

Dataset scale is calculated from processed data rather than guessed in advance. Model training uses temporal vintage splits and excludes configured terminal and loss fields from features. Scenario outputs are meant to be clear research sensitivities, not rating-agency models, investment advice, or a cash-flow waterfall.

See [architecture](docs/architecture.md), [methodology](docs/methodology.md), [validation rules](docs/validation_rules.md), and [limitations](docs/limitations.md).

## Freddie Mac dataset access and setup

The Freddie Mac Single-Family Loan-Level Dataset is available through the official [dataset resource page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset). Register, accept the applicable terms of use, and download the permitted annual or sample archives. The source data is licensed and restricted, so do not redistribute it or commit it to Git.

For a practical local build, download the origination and performance files for the desired vintages (for example, the 2012-2022 samples). Extract each archive and place the unmodified files in vintage folders:

```text
data/raw/2012/sample_orig_2012.txt
data/raw/2012/sample_svcg_2012.txt
data/raw/2013/sample_orig_2013.txt
data/raw/2013/sample_svcg_2013.txt
...
```

Ingestion detects the vintage from the file name and records source metadata, file size, checksum, and ingestion timestamp. See [data/README.md](data/README.md) for directory and licensing details.

## Windows commands (without Make)

GNU Make is optional and is not installed by default on Windows. Run these PowerShell commands from the repository root instead:

```powershell
cd C:\Users\abhis\projects\StrataCredit
py -3.13 -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"

# Synthetic fixture and automated test suite
py -3.13 -m stratacredit.fixtures.generate
py -3.13 -m pytest tests -m "not integration" --no-cov -q

# Raw data to validated Bronze, Silver, and Gold tables
py -3.13 -m stratacredit.ingest.runner
py -3.13 -m stratacredit.quality.runner
py -3.13 -m stratacredit.transform.runner
py -3.13 -m stratacredit.analytics.runner

# Models, scoring, pool selection, stress, and reporting
py -3.13 -m stratacredit.models.full_trainer
py -3.13 -m stratacredit.models.xgb_trainer
py -3.13 -m stratacredit.models.scoring
py -3.13 -m stratacredit.models.mlflow_tracking
py -3.13 -m stratacredit.models.leakage_audit
py -3.13 -m stratacredit.optimization.runner
py -3.13 -m stratacredit.scenarios.runner
py -3.13 -m stratacredit.reporting.runner

# Analyst workbench
py -3.13 -m streamlit run app\streamlit_app.py
```

## Analyst workbench pages

| Page | Purpose |
| --- | --- |
| Executive Deal Snapshot | Active-universe UPB, weighted-average collateral metrics, delinquency, concentrations, and reference constraint comparison. |
| Collateral Stratification | Filterable FICO, LTV, DTI, coupon, vintage, geography, occupancy, and purpose distributions. |
| Historical Performance | CPR, delinquency transitions, credit events, recoveries, and loss severity. Interactive panel charts use a deterministic sample to stay responsive. |
| Pool Builder | Eligibility filtering, risk-aware selection, constraint reconciliation, candidate-versus-selected analysis, and selected-pool/repline downloads. |
| Model & Backtest | Out-of-time metrics, cohort backtests, feature importance, leakage review, and local MLflow experiments. |
| Data Quality | Rule-level validations, warnings, quarantined records, and source-to-target reconciliation. |

## Pool-selection and model limitations

Pool selection is a transparent, deterministic greedy process. It applies the configured balance, collateral-quality, concentration, and state-UPB guardrails, then reports every constraint as PASS or FAIL. A PASS result means the configured numerical limits were met. It does not prove that the pool is globally optimal or suitable for a particular investment or transaction.

Models are statistical estimates conditioned on the disclosed Freddie Mac population, historical periods, and project feature definitions. Temporal validation and leakage checks reduce but cannot eliminate model risk. Credit-event models using current or trailing delinquency are servicing-risk tools and should not be interpreted as clean-pool ratings. Stress results are illustrative multiplier-based sensitivities, not Fitch, KBRA, Moody's, S&P, or any other rating-agency methodology, and StrataCredit is not a full MBS cash-flow waterfall.

> StrataCredit is for research and informational purposes only. It is not investment advice, an offer or solicitation, a credit rating, or a proprietary rating-agency model.

## Author

Built by [Abhishek Rithik Origanti](https://github.com/abhioriganti). For questions or feedback, contact [abhishekoriganti@gmail.com](mailto:abhishekoriganti@gmail.com).
