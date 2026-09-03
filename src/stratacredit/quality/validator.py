"""Validation engine: applies rules to DataFrames and records results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import polars as pl

from stratacredit.quality.rules import Severity, ValidationRule


@dataclass
class ValidationResult:
    run_id: str
    dataset: str
    rule_id: str
    description: str
    severity: str
    rows_checked: int
    rows_failed: int
    failure_rate: float
    status: str  # PASS / FAIL / QUARANTINE / WARN
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class QuarantineRecord:
    run_id: str
    dataset: str
    rule_id: str
    reason: str
    row_index: int
    loan_id: str | None
    reporting_period: str | None
    source_file: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


def apply_rule(
    df: pl.DataFrame,
    rule: ValidationRule,
    run_id: str,
) -> tuple[ValidationResult, list[QuarantineRecord]]:
    """Apply a single validation rule to a DataFrame.

    Args:
        df: DataFrame to validate.
        rule: The validation rule to apply.
        run_id: Unique run identifier.

    Returns:
        (ValidationResult, list[QuarantineRecord])
    """
    passed: pl.Series = rule.check_fn(df)
    n_checked = len(df)
    n_failed = int((~passed).sum())
    failure_rate = n_failed / n_checked if n_checked > 0 else 0.0

    if n_failed == 0:
        status = "PASS"
    elif rule.severity == Severity.FAIL:
        status = "FAIL"
    elif rule.severity == Severity.QUARANTINE:
        status = "QUARANTINE"
    else:
        status = "WARN"

    result = ValidationResult(
        run_id=run_id,
        dataset=rule.dataset,
        rule_id=rule.rule_id,
        description=rule.description,
        severity=rule.severity.value,
        rows_checked=n_checked,
        rows_failed=n_failed,
        failure_rate=failure_rate,
        status=status,
    )

    # Build quarantine records for individual failed rows
    quarantine: list[QuarantineRecord] = []
    if n_failed > 0 and rule.severity in (Severity.QUARANTINE, Severity.FAIL):
        failed_df = df.filter(~passed)
        loan_id_col = "loan_id" if "loan_id" in df.columns else None
        rp_col = "reporting_period" if "reporting_period" in df.columns else None
        src_col = "_source_file" if "_source_file" in df.columns else None

        for i, row in enumerate(failed_df.iter_rows(named=True)):
            quarantine.append(
                QuarantineRecord(
                    run_id=run_id,
                    dataset=rule.dataset,
                    rule_id=rule.rule_id,
                    reason=rule.description,
                    row_index=i,
                    loan_id=str(row[loan_id_col]) if loan_id_col else None,
                    reporting_period=str(row[rp_col]) if rp_col else None,
                    source_file=str(row[src_col]) if src_col else None,
                )
            )

    return result, quarantine


def run_validation(
    df: pl.DataFrame,
    rules: list[ValidationRule],
    run_id: str | None = None,
) -> tuple[list[ValidationResult], list[QuarantineRecord]]:
    """Run all validation rules against a DataFrame."""
    run_id = run_id or str(uuid.uuid4())
    all_results: list[ValidationResult] = []
    all_quarantine: list[QuarantineRecord] = []

    for rule in rules:
        result, quarantine = apply_rule(df, rule, run_id)
        all_results.append(result)
        all_quarantine.extend(quarantine)

    return all_results, all_quarantine


def results_to_dataframe(results: list[ValidationResult]) -> pl.DataFrame:
    """Convert validation results to a Polars DataFrame."""
    if not results:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "run_id": r.run_id,
                "dataset": r.dataset,
                "rule_id": r.rule_id,
                "description": r.description,
                "severity": r.severity,
                "rows_checked": r.rows_checked,
                "rows_failed": r.rows_failed,
                "failure_rate": r.failure_rate,
                "status": r.status,
                "timestamp": str(r.timestamp),
            }
            for r in results
        ]
    )


def quarantine_to_dataframe(records: list[QuarantineRecord]) -> pl.DataFrame:
    """Convert quarantine records to a Polars DataFrame."""
    if not records:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "dataset": pl.Utf8,
                "rule_id": pl.Utf8,
                "reason": pl.Utf8,
                "row_index": pl.Int64,
                "loan_id": pl.Utf8,
                "reporting_period": pl.Utf8,
                "source_file": pl.Utf8,
                "timestamp": pl.Utf8,
            }
        )
    return pl.DataFrame(
        [
            {
                "run_id": r.run_id,
                "dataset": r.dataset,
                "rule_id": r.rule_id,
                "reason": r.reason,
                "row_index": r.row_index,
                "loan_id": r.loan_id,
                "reporting_period": r.reporting_period,
                "source_file": r.source_file,
                "timestamp": str(r.timestamp),
            }
            for r in records
        ]
    )


def persist_validation_results(
    results: list[ValidationResult], records: list[QuarantineRecord]
) -> None:
    """Append auditable rule results and record-level quarantines to DuckDB."""
    from stratacredit.db import get_connection

    result_df = results_to_dataframe(results)
    quarantine_df = quarantine_to_dataframe(records)
    with get_connection("silver") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS silver.data_quality_results (
              run_id VARCHAR, dataset VARCHAR, rule_id VARCHAR, description VARCHAR,
              severity VARCHAR, rows_checked BIGINT, rows_failed BIGINT,
              failure_rate DOUBLE, status VARCHAR, timestamp VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS silver.quarantine_records (
              run_id VARCHAR, dataset VARCHAR, rule_id VARCHAR, reason VARCHAR,
              row_index BIGINT, loan_id VARCHAR, reporting_period VARCHAR,
              source_file VARCHAR, timestamp VARCHAR
            )
        """)
        if not result_df.is_empty():
            conn.register("dq_results_frame", result_df.to_pandas())
            conn.execute("INSERT INTO silver.data_quality_results SELECT * FROM dq_results_frame")
        if not quarantine_df.is_empty():
            conn.register("dq_quarantine_frame", quarantine_df.to_pandas())
            conn.execute("INSERT INTO silver.quarantine_records SELECT * FROM dq_quarantine_frame")
