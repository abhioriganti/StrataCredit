# Validation Rules

## Severity Levels

| Severity | Meaning |
|----------|---------|
| FAIL | Critical error. Pipeline cannot continue. |
| QUARANTINE | Individual records excluded from analytics but retained with reason. |
| WARN | Potential anomaly surfaced. Records remain in analytical tables. |

## Origination Rules

| Rule ID | Severity | Check |
|---------|----------|-------|
| ORIG_001 | FAIL | `loan_id` must not be null |
| ORIG_002 | QUARANTINE | `original_upb` must be positive |
| ORIG_003 | WARN | `fico` must be in range 300–850 if present |
| ORIG_004 | WARN | `original_ltv` must be in range 1–200 if present |
| ORIG_005 | WARN | `original_cltv` must be in range 1–200 if present |
| ORIG_006 | WARN | `dti` must be in range 0–100 if present |
| ORIG_007 | WARN | `original_interest_rate` must be in range 0.5–20 if present |
| ORIG_008 | QUARANTINE | `origination_date` must not be null |
| ORIG_009 | WARN | `original_loan_term` must be positive if present |

## Performance Rules

| Rule ID | Severity | Check |
|---------|----------|-------|
| PERF_001 | FAIL | `loan_id` must not be null |
| PERF_002 | FAIL | `reporting_period` must not be null |
| PERF_003 | QUARANTINE | `current_upb` must be ≥ 0 |
| PERF_004 | WARN | `loan_age` must be ≥ 0 if present |
| PERF_005 | WARN | `current_interest_rate` must be in range 0–20 if present |
| PERF_006 | WARN | `delinquency_months` must be 0–99 if present |
| PERF_007 | WARN | `actual_loss` must be ≥ 0 if present |

## Leakage Guard Rules

Applied to model snapshot DataFrames before training.

| Rule ID | Severity | Check |
|---------|----------|-------|
| LEAK_001 | FAIL | `zero_balance_code` must not appear as a feature column |
| LEAK_002 | FAIL | `actual_loss` must not appear as a feature column |
| LEAK_003 | FAIL | `net_sale_proceeds` must not appear as a feature column |

## Reconciliation Checks

| Check | Tolerance | Description |
|-------|-----------|-------------|
| origination_vs_panel_unique_loans | < 1% diff | Unique loans in origination vs panel |
| performance_orphan_loan_ids | = 0 | Performance records with no origination match |
| performance_vs_panel_total_upb | < 0.1% diff | Total UPB must match between layers |
