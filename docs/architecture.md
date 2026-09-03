# Architecture

## System Overview

StrataCredit is a structured-finance analytics platform that converts raw
mortgage loan-level data into validated analytical datasets, collateral
analysis, historical performance metrics, predictive models, constrained
asset-pool selection, stress scenarios, and deal-facing outputs.

```
Raw Mortgage Data (Freddie Mac Single-Family Loan-Level Dataset)
        ↓
Ingestion & Bronze Parquet (src/stratacredit/ingest/)
        ↓
Data Validation & Reconciliation (src/stratacredit/quality/)
        ↓
Silver Layer: Canonical Loan-Month Panel (src/stratacredit/transform/)
        ↓
Gold Layer: Analytical Tables (src/stratacredit/analytics/)
        ↓
Performance Analytics (src/stratacredit/performance/)
        ↓
Feature Engineering (src/stratacredit/features/)
        ↓
Predictive Models + MLflow (src/stratacredit/models/)
        ↓
Pool Selection (src/stratacredit/optimization/)
        ↓
Stress Scenarios (src/stratacredit/scenarios/)
        ↓
Reports + Exports (src/stratacredit/reporting/)
        ↓
Streamlit Analyst Workbench (app/)
```

## Data Layers

### RAW
Original downloaded Freddie Mac text files. Never modified.
Stored in `data/raw/`. Git-ignored. File checksums recorded on ingest.

### BRONZE
Typed Parquet files produced by the ingest module.
One Parquet file per source file, partitioned by `file_type/vintage/`.
Contains all source columns plus metadata columns (`_source_file`,
`_source_vintage`, `_ingested_at`, `_file_checksum`).

### SILVER
Standardized analytical tables in DuckDB (`data/silver/stratacredit_silver.duckdb`).

- `silver.loan_origination` — one record per mortgage
- `silver.loan_performance` — one record per loan per month
- `silver.loan_month_panel` — canonical join: one row = one loan × one reporting month

The panel is the primary analytical dataset for all downstream work.

### GOLD
Analyst-facing aggregated tables in DuckDB (`data/gold/stratacredit_gold.duckdb`).
Built from the Silver panel by the Gold SQL scripts in `sql/gold/`.

## Module Design Principles

1. **No circular imports**: each module imports only from lower layers.
2. **Config-driven**: all analytical parameters live in `configs/`.
3. **Leakage-safe**: the features module enforces forbidden-column lists.
4. **Idempotent ingestion**: checksums prevent double-processing.
5. **Reproduced from fixtures**: all tests use synthetic data; no real data in CI.

## Key Design Decisions

### DuckDB vs PostgreSQL
DuckDB is used for analytical queries because it reads Parquet directly,
requires no server process, and handles the full dataset efficiently.
PostgreSQL is optional (listed in `pyproject.toml[api]`) but not required.

### Polars vs pandas
Polars is the primary DataFrame library for performance and memory efficiency.
pandas is used only where required by sklearn/MLflow APIs.

### Greedy Pool Selection (V1)
V1 uses deterministic greedy constrained selection rather than a full MIP.
This runs fast on any dataset size without OR-Tools/PuLP.
V2 MIP support is wired in `pool_constraints.yaml` (`use_mip: false`).

### Temporal Validation
Models use origination-vintage splits (train 2012–2017, val 2018–2019,
test 2020–2022). Random splits are never used as the primary evaluation.
