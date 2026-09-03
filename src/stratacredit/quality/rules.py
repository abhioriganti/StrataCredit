"""Data quality validation rules.

Defines all validation rules applied to Bronze and Silver layers.
Each rule has a severity: FAIL, QUARANTINE, or WARN.

FAIL        - Critical error. Pipeline should stop.
QUARANTINE  - Individual records excluded but pipeline continues.
WARN        - Anomaly surfaced but records remain usable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import polars as pl


class Severity(str, Enum):
    FAIL = "FAIL"
    QUARANTINE = "QUARANTINE"
    WARN = "WARN"


@dataclass
class ValidationRule:
    rule_id: str
    dataset: str
    description: str
    severity: Severity
    check_fn: Callable[[pl.DataFrame], pl.Series]
    # check_fn returns a boolean Series: True = PASS, False = FAIL


# ── Helper predicates ──────────────────────────────────────────────────────────


def _not_null(col: str) -> Callable[[pl.DataFrame], pl.Series]:
    return lambda df: df[col].is_not_null() if col in df.columns else pl.Series([True] * len(df))


def _in_range(col: str, lo: float, hi: float) -> Callable[[pl.DataFrame], pl.Series]:
    def _check(df: pl.DataFrame) -> pl.Series:
        if col not in df.columns:
            return pl.Series([True] * len(df))
        c = df[col]
        return c.is_null() | ((c >= lo) & (c <= hi))

    return _check


def _non_negative(col: str) -> Callable[[pl.DataFrame], pl.Series]:
    def _check(df: pl.DataFrame) -> pl.Series:
        if col not in df.columns:
            return pl.Series([True] * len(df))
        c = df[col]
        return c.is_null() | (c >= 0)

    return _check


# ── Origination Rules ─────────────────────────────────────────────────────────

ORIGINATION_RULES: list[ValidationRule] = [
    ValidationRule(
        rule_id="ORIG_001",
        dataset="loan_origination",
        description="loan_id must not be null",
        severity=Severity.FAIL,
        check_fn=_not_null("loan_id"),
    ),
    ValidationRule(
        rule_id="ORIG_002",
        dataset="loan_origination",
        description="original_upb must be positive",
        severity=Severity.QUARANTINE,
        check_fn=lambda df: df["original_upb"].is_null() | (df["original_upb"] > 0),
    ),
    ValidationRule(
        rule_id="ORIG_003",
        dataset="loan_origination",
        description="FICO must be in range 300–850 if present",
        severity=Severity.WARN,
        check_fn=_in_range("fico", 300, 850),
    ),
    ValidationRule(
        rule_id="ORIG_004",
        dataset="loan_origination",
        description="LTV must be in range 1–200 if present",
        severity=Severity.WARN,
        check_fn=_in_range("original_ltv", 1, 200),
    ),
    ValidationRule(
        rule_id="ORIG_005",
        dataset="loan_origination",
        description="CLTV must be in range 1–200 if present",
        severity=Severity.WARN,
        check_fn=_in_range("original_cltv", 1, 200),
    ),
    ValidationRule(
        rule_id="ORIG_006",
        dataset="loan_origination",
        description="DTI must be in range 0–100 if present",
        severity=Severity.WARN,
        check_fn=_in_range("dti", 0, 100),
    ),
    ValidationRule(
        rule_id="ORIG_007",
        dataset="loan_origination",
        description="original_interest_rate must be in range 0.5–20 if present",
        severity=Severity.WARN,
        check_fn=_in_range("original_interest_rate", 0.5, 20.0),
    ),
    ValidationRule(
        rule_id="ORIG_008",
        dataset="loan_origination",
        description="origination_date must not be null",
        severity=Severity.QUARANTINE,
        check_fn=_not_null("origination_date"),
    ),
    ValidationRule(
        rule_id="ORIG_009",
        dataset="loan_origination",
        description="original_loan_term must be positive if present",
        severity=Severity.WARN,
        check_fn=lambda df: (
            df["original_loan_term"].is_null() | (df["original_loan_term"] > 0)
            if "original_loan_term" in df.columns
            else pl.Series([True] * len(df))
        ),
    ),
]

# ── Performance Rules ─────────────────────────────────────────────────────────

PERFORMANCE_RULES: list[ValidationRule] = [
    ValidationRule(
        rule_id="PERF_001",
        dataset="loan_performance",
        description="loan_id must not be null",
        severity=Severity.FAIL,
        check_fn=_not_null("loan_id"),
    ),
    ValidationRule(
        rule_id="PERF_002",
        dataset="loan_performance",
        description="reporting_period must not be null",
        severity=Severity.FAIL,
        check_fn=_not_null("reporting_period"),
    ),
    ValidationRule(
        rule_id="PERF_003",
        dataset="loan_performance",
        description="current_upb must be >= 0",
        severity=Severity.QUARANTINE,
        check_fn=_non_negative("current_upb"),
    ),
    ValidationRule(
        rule_id="PERF_004",
        dataset="loan_performance",
        description="loan_age must be >= 0 if present",
        severity=Severity.WARN,
        check_fn=_non_negative("loan_age"),
    ),
    ValidationRule(
        rule_id="PERF_005",
        dataset="loan_performance",
        description="current_interest_rate must be in range 0–20 if present",
        severity=Severity.WARN,
        check_fn=_in_range("current_interest_rate", 0, 20),
    ),
    ValidationRule(
        rule_id="PERF_006",
        dataset="loan_performance",
        description="delinquency_months must be 0–99 if present",
        severity=Severity.WARN,
        check_fn=_in_range("delinquency_months", 0, 99),
    ),
    ValidationRule(
        rule_id="PERF_007",
        dataset="loan_performance",
        description="loss_severity components should be non-negative if present",
        severity=Severity.WARN,
        check_fn=lambda df: (
            df["actual_loss"].cast(pl.Float64, strict=False).is_null()
            | (df["actual_loss"].cast(pl.Float64, strict=False) >= 0)
            if "actual_loss" in df.columns
            else pl.Series([True] * len(df))
        ),
    ),
]

# ── Panel Leakage Guard Rules ─────────────────────────────────────────────────

PANEL_LEAKAGE_RULES: list[ValidationRule] = [
    ValidationRule(
        rule_id="LEAK_001",
        dataset="model_snapshot",
        description="zero_balance_code must not appear as a feature column",
        severity=Severity.FAIL,
        check_fn=lambda df: pl.Series(
            [False] * len(df) if "zero_balance_code" in df.columns else [True] * len(df)
        ),
    ),
    ValidationRule(
        rule_id="LEAK_002",
        dataset="model_snapshot",
        description="actual_loss must not appear as a feature column",
        severity=Severity.FAIL,
        check_fn=lambda df: pl.Series(
            [False] * len(df) if "actual_loss" in df.columns else [True] * len(df)
        ),
    ),
    ValidationRule(
        rule_id="LEAK_003",
        dataset="model_snapshot",
        description="net_sale_proceeds must not appear as a feature column",
        severity=Severity.FAIL,
        check_fn=lambda df: pl.Series(
            [False] * len(df) if "net_sale_proceeds" in df.columns else [True] * len(df)
        ),
    ),
]

ALL_RULES: list[ValidationRule] = ORIGINATION_RULES + PERFORMANCE_RULES + PANEL_LEAKAGE_RULES
