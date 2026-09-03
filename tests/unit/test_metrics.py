"""Unit tests for portfolio metrics and weighted average calculations.

These tests verify core financial calculations against known expected values.
All inputs and expected outputs are hand-computed.
"""

import polars as pl
import pytest

from stratacredit.analytics.metrics import compute_portfolio_metrics, weighted_average


@pytest.mark.unit
class TestWeightedAverage:
    """Tests for the weighted_average() function."""

    def test_simple_equal_weights(self):
        df = pl.DataFrame({"value": [2.0, 4.0, 6.0], "current_upb": [1.0, 1.0, 1.0]})
        result = weighted_average(df, "value")
        assert result == pytest.approx(4.0, rel=1e-6)

    def test_upb_weighted(self):
        # Loan A: value=3, upb=100; Loan B: value=7, upb=100 → WA = 5.0
        df = pl.DataFrame({"value": [3.0, 7.0], "current_upb": [100.0, 100.0]})
        assert weighted_average(df, "value") == pytest.approx(5.0)

    def test_unequal_weights(self):
        # value=2 with weight=1, value=4 with weight=3 → WA = (2+12)/4 = 3.5
        df = pl.DataFrame({"value": [2.0, 4.0], "current_upb": [1.0, 3.0]})
        assert weighted_average(df, "value") == pytest.approx(3.5)

    def test_null_values_excluded(self):
        # Null rows should not affect result
        df = pl.DataFrame({"value": [None, 4.0, 6.0], "current_upb": [100.0, 100.0, 100.0]})
        result = weighted_average(df, "value")
        assert result == pytest.approx(5.0)

    def test_all_null_returns_none(self):
        df = pl.DataFrame({"value": [None, None], "current_upb": [1.0, 1.0]})
        assert weighted_average(df, "value") is None

    def test_missing_column_returns_none(self):
        df = pl.DataFrame({"other": [1.0], "current_upb": [1.0]})
        assert weighted_average(df, "missing_col") is None

    def test_zero_weight_returns_none(self):
        df = pl.DataFrame({"value": [5.0], "current_upb": [0.0]})
        assert weighted_average(df, "value") is None


@pytest.mark.unit
class TestPortfolioMetrics:
    """Tests for compute_portfolio_metrics()."""

    def _make_df(self, n: int = 4) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "loan_id": [f"L{i}" for i in range(n)],
                "current_upb": [200_000.0, 150_000.0, 300_000.0, 250_000.0],
                "current_interest_rate": [4.0, 3.5, 4.5, 5.0],
                "fico": [720, 740, 680, 760],
                "original_ltv": [75.0, 70.0, 80.0, 65.0],
                "original_cltv": [78.0, 72.0, 83.0, 67.0],
                "dti": [35.0, 30.0, 40.0, 28.0],
                "loan_age": [12, 24, 6, 36],
                "remaining_term": [348, 336, 354, 324],
                "loan_purpose": ["P", "R", "P", "C"],
                "occupancy": ["P", "P", "I", "P"],
                "property_state": ["CA", "TX", "FL", "CA"],
                "mortgage_insurance_pct": [None, None, 0.5, None],
                "delinquency_state": ["CURRENT", "CURRENT", "DQ30", "CURRENT"],
                "modification_flag": [False, False, False, False],
            }
        )

    def test_loan_count(self):
        m = compute_portfolio_metrics(self._make_df())
        assert m.loan_count == 4

    def test_total_upb(self):
        m = compute_portfolio_metrics(self._make_df())
        assert m.total_current_upb == pytest.approx(900_000.0)

    def test_wa_fico(self):
        df = self._make_df()
        # Hand-compute: (720*200k + 740*150k + 680*300k + 760*250k) / 900k
        expected = (720 * 200_000 + 740 * 150_000 + 680 * 300_000 + 760 * 250_000) / 900_000
        m = compute_portfolio_metrics(df)
        assert m.wa_fico == pytest.approx(expected, rel=1e-4)

    def test_investor_pct(self):
        m = compute_portfolio_metrics(self._make_df())
        # 1 investor loan with upb=300k out of 900k total
        assert m.investor_pct == pytest.approx(300_000 / 900_000 * 100, rel=1e-4)

    def test_purchase_pct(self):
        m = compute_portfolio_metrics(self._make_df())
        # 2 purchase loans: 200k + 300k = 500k
        assert m.purchase_pct == pytest.approx(500_000 / 900_000 * 100, rel=1e-4)

    def test_dq30_pct(self):
        m = compute_portfolio_metrics(self._make_df())
        # 1 DQ30 loan with upb=300k
        assert m.dq30_pct == pytest.approx(300_000 / 900_000 * 100, rel=1e-4)

    def test_largest_state_is_ca(self):
        m = compute_portfolio_metrics(self._make_df())
        # CA: 200k + 250k = 450k
        assert m.largest_state == "CA"
        assert m.largest_state_pct == pytest.approx(450_000 / 900_000 * 100, rel=1e-4)

    def test_empty_df_returns_zeroes(self):
        df = pl.DataFrame(
            {
                "current_upb": pl.Series([], dtype=pl.Float64),
                "loan_id": pl.Series([], dtype=pl.Utf8),
            }
        )
        m = compute_portfolio_metrics(df)
        assert m.loan_count == 0
        assert m.total_current_upb == 0.0
