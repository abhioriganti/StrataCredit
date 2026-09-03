"""Data tests: schema, uniqueness, leakage guards."""

import polars as pl
import pytest

from stratacredit.quality.rules import (
    ORIGINATION_RULES,
    PANEL_LEAKAGE_RULES,
    PERFORMANCE_RULES,
)
from stratacredit.quality.validator import run_validation


def _orig_df(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "origination_date": ["2020-01-01"] * n,
            "original_upb": [200_000.0] * n,
            "fico": [740] * n,
            "original_ltv": [75.0] * n,
            "original_cltv": [77.0] * n,
            "dti": [35.0] * n,
            "original_interest_rate": [4.0] * n,
            "original_loan_term": [360] * n,
        }
    )


def _perf_df(n: int = 10) -> pl.DataFrame:
    from datetime import date

    return pl.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "reporting_period": [date(2020, 1, 1)] * n,
            "current_upb": [199_000.0] * n,
            "loan_age": [1] * n,
            "current_interest_rate": [4.0] * n,
            "delinquency_months": [0] * n,
            "actual_loss": [None] * n,
        }
    )


@pytest.mark.data
class TestOriginationValidation:
    def test_valid_data_passes(self):
        results, _ = run_validation(_orig_df(), ORIGINATION_RULES)
        for r in results:
            assert r.status in ("PASS", "WARN"), f"{r.rule_id} unexpectedly failed"

    def test_null_loan_id_triggers_fail(self):
        df = _orig_df().with_columns(
            pl.when(pl.col("loan_id") == "L0")
            .then(None)
            .otherwise(pl.col("loan_id"))
            .alias("loan_id")
        )
        results, _ = run_validation(df, ORIGINATION_RULES)
        assert any(r.rule_id == "ORIG_001" and r.status == "FAIL" for r in results)

    def test_negative_upb_triggers_quarantine(self):
        df = _orig_df().with_columns(
            pl.when(pl.col("loan_id") == "L0")
            .then(pl.lit(-1.0))
            .otherwise(pl.col("original_upb"))
            .alias("original_upb")
        )
        results, quarantine = run_validation(df, ORIGINATION_RULES)
        assert any(r.rule_id == "ORIG_002" and r.status == "QUARANTINE" for r in results)
        assert len(quarantine) > 0

    def test_fico_out_of_range_is_warn(self):
        df = _orig_df().with_columns(
            pl.when(pl.col("loan_id") == "L0")
            .then(pl.lit(999.0))
            .otherwise(pl.col("fico").cast(pl.Float64))
            .alias("fico")
        )
        results, _ = run_validation(df, ORIGINATION_RULES)
        r = next(x for x in results if x.rule_id == "ORIG_003")
        assert r.status == "WARN"
        assert r.rows_failed == 1


@pytest.mark.data
class TestPerformanceValidation:
    def test_valid_data_passes(self):
        results, _ = run_validation(_perf_df(), PERFORMANCE_RULES)
        for r in results:
            assert r.status in ("PASS", "WARN")

    def test_negative_upb_quarantined(self):
        df = _perf_df().with_columns(
            pl.when(pl.col("loan_id") == "L0")
            .then(pl.lit(-1.0))
            .otherwise(pl.col("current_upb"))
            .alias("current_upb")
        )
        results, _ = run_validation(df, PERFORMANCE_RULES)
        assert any(r.rule_id == "PERF_003" and r.rows_failed > 0 for r in results)


@pytest.mark.data
class TestLeakageGuards:
    def test_zero_balance_code_triggers_fail(self):
        df = pl.DataFrame({"loan_id": ["L1"], "zero_balance_code": ["01"]})
        results, _ = run_validation(df, PANEL_LEAKAGE_RULES)
        assert any(r.rule_id == "LEAK_001" and r.status == "FAIL" for r in results)

    def test_actual_loss_triggers_fail(self):
        df = pl.DataFrame({"loan_id": ["L1"], "actual_loss": [5000.0]})
        results, _ = run_validation(df, PANEL_LEAKAGE_RULES)
        assert any(r.rule_id == "LEAK_002" and r.status == "FAIL" for r in results)

    def test_clean_feature_df_passes(self):
        df = pl.DataFrame({"loan_id": ["L1"], "fico": [740], "original_ltv": [75.0]})
        results, _ = run_validation(df, PANEL_LEAKAGE_RULES)
        assert all(r.status == "PASS" for r in results)
