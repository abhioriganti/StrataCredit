"""Freddie Mac source schema definitions.

Column name lists and Polars dtype mappings for origination
and performance files.  Derived from data.yaml but expressed
as Python objects for use in the Polars scanning code.
"""

from __future__ import annotations

import polars as pl

# ── Origination columns (index-ordered, pipe-delimited, no header) ────────────
ORIG_SCHEMA: list[tuple[str, pl.PolarsDataType]] = [
    ("credit_score", pl.Int32),
    ("first_payment_date", pl.Utf8),
    ("first_time_homebuyer_flag", pl.Utf8),
    ("maturity_date", pl.Utf8),
    ("msa", pl.Utf8),
    ("mi_pct", pl.Float64),
    ("num_units", pl.Int32),
    ("occupancy_status", pl.Utf8),
    ("cltv", pl.Float64),
    ("dti", pl.Float64),
    ("original_upb", pl.Float64),
    ("ltv", pl.Float64),
    ("original_interest_rate", pl.Float64),
    ("channel", pl.Utf8),
    ("prepayment_penalty_flag", pl.Utf8),
    ("amort_type", pl.Utf8),
    ("property_state", pl.Utf8),
    ("property_type", pl.Utf8),
    ("postal_code", pl.Utf8),
    ("loan_sequence_number", pl.Utf8),
    ("loan_purpose", pl.Utf8),
    ("original_loan_term", pl.Int32),
    ("num_borrowers", pl.Int32),
    ("seller_name", pl.Utf8),
    ("servicer_name", pl.Utf8),
    ("super_conforming_flag", pl.Utf8),
    ("pre_harp_sequence_number", pl.Utf8),
    ("program_indicator", pl.Utf8),
    ("harp_indicator", pl.Utf8),
    ("property_valuation_method", pl.Utf8),
    ("interest_only_indicator", pl.Utf8),
    ("mi_cancellation_indicator", pl.Utf8),
]

ORIG_COLUMNS = [name for name, _ in ORIG_SCHEMA]
ORIG_DTYPE_MAP = dict(ORIG_SCHEMA)

# Sentinel values used by Freddie Mac for missing data
ORIG_SENTINEL_INT_COLS = ["credit_score", "num_units", "original_loan_term", "num_borrowers"]
ORIG_SENTINEL_FLOAT_COLS = [
    "mi_pct",
    "cltv",
    "dti",
    "original_upb",
    "ltv",
    "original_interest_rate",
]
SENTINEL_INT_VALUES = [9, 99, 999, 9999]
SENTINEL_FLOAT_VALUES = [9.0, 99.0, 999.0, 9999.0]

# ── Performance columns ────────────────────────────────────────────────────────
PERF_SCHEMA: list[tuple[str, pl.PolarsDataType]] = [
    ("loan_sequence_number", pl.Utf8),
    ("monthly_reporting_period", pl.Utf8),
    ("current_actual_upb", pl.Float64),
    ("current_loan_delinquency_status", pl.Utf8),
    ("loan_age", pl.Int32),
    ("remaining_months_to_legal_maturity", pl.Int32),
    ("defect_settlement_date", pl.Utf8),
    ("modification_flag", pl.Utf8),
    ("zero_balance_code", pl.Utf8),
    ("zero_balance_effective_date", pl.Utf8),
    ("current_interest_rate", pl.Float64),
    ("current_deferred_upb", pl.Float64),
    ("ddlpi", pl.Utf8),
    ("mi_recoveries", pl.Float64),
    ("net_sale_proceeds", pl.Utf8),
    ("non_mi_recoveries", pl.Float64),
    ("expenses", pl.Float64),
    ("legal_costs", pl.Float64),
    ("maintenance_costs", pl.Float64),
    ("taxes_insurance", pl.Float64),
    ("miscellaneous_expenses", pl.Float64),
    ("actual_loss_calculation", pl.Float64),
    ("modification_cost", pl.Float64),
    ("step_modification_flag", pl.Utf8),
    ("deferred_payment_plan", pl.Utf8),
    ("estimated_ltv", pl.Float64),
    ("zero_balance_removal_upb", pl.Float64),
    ("delinquent_accrued_interest", pl.Float64),
    ("delinquency_due_to_disaster", pl.Utf8),
    ("borrower_assistance_status_code", pl.Utf8),
    ("current_month_modification_cost", pl.Float64),
    ("interest_bearing_upb", pl.Float64),
]

PERF_COLUMNS = [name for name, _ in PERF_SCHEMA]
PERF_DTYPE_MAP = dict(PERF_SCHEMA)

PERF_SENTINEL_INT_COLS = ["loan_age", "remaining_months_to_legal_maturity"]
PERF_SENTINEL_FLOAT_COLS = [
    "current_actual_upb",
    "current_interest_rate",
    "current_deferred_upb",
    "estimated_ltv",
    "mi_recoveries",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_costs",
    "taxes_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "modification_cost",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "current_month_modification_cost",
    "interest_bearing_upb",
]
