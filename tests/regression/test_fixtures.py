"""Golden fixture regression tests.

Generates synthetic fixture data and verifies that computed metrics
match expected values derived from the seed=42 deterministic generator.

These tests must pass every run without modification.
If expected values change, the fixture generator logic changed — investigate.
"""

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="module")
def fixture_data():
    """Generate fixture data once per test module."""
    from stratacredit.fixtures.generate import run

    result = run(output_dir=Path("data/fixtures"))
    orig = pl.read_parquet(result["origination_path"])
    perf = pl.read_parquet(result["performance_path"])
    return orig, perf, result


@pytest.mark.regression
class TestFixtureShape:
    def test_loan_count(self, fixture_data):
        orig, _, meta = fixture_data
        assert meta["n_loans"] == 500

    def test_panel_rows_positive(self, fixture_data):
        _, perf, meta = fixture_data
        assert meta["n_panel_rows"] > 1000

    def test_origination_has_required_columns(self, fixture_data):
        orig, _, _ = fixture_data
        required = {
            "loan_sequence_number",
            "original_upb",
            "original_interest_rate",
            "ltv",
            "credit_score",
            "dti",
            "loan_purpose",
            "occupancy_status",
            "property_state",
            "original_loan_term",
        }
        assert required.issubset(set(orig.columns))

    def test_performance_has_required_columns(self, fixture_data):
        _, perf, _ = fixture_data
        required = {
            "loan_sequence_number",
            "monthly_reporting_period",
            "current_actual_upb",
            "current_loan_delinquency_status",
            "loan_age",
            "zero_balance_code",
        }
        assert required.issubset(set(perf.columns))

    def test_no_duplicate_origination_ids(self, fixture_data):
        orig, _, _ = fixture_data
        assert orig["loan_sequence_number"].n_unique() == len(orig)

    def test_all_vintages_present(self, fixture_data):
        orig, _, _ = fixture_data
        # Fixtures cover 2018–2022
        vintages = set(orig["_source_vintage"].unique().to_list())
        for y in ["2018", "2019", "2020", "2021", "2022"]:
            assert y in vintages

    def test_upb_values_are_positive(self, fixture_data):
        orig, _, _ = fixture_data
        assert (orig["original_upb"] > 0).all()

    def test_performance_upb_non_negative(self, fixture_data):
        _, perf, _ = fixture_data
        upb_col = perf["current_actual_upb"].drop_nulls()
        assert (upb_col >= 0).all()


@pytest.mark.regression
class TestFixtureMetrics:
    def test_wa_fico_in_plausible_range(self, fixture_data):
        orig, _, _ = fixture_data
        upb = orig["original_upb"]
        fico = orig["credit_score"].cast(pl.Float64)
        valid = orig.filter(fico.is_not_null() & (upb > 0))
        wa_fico = (valid["credit_score"].cast(pl.Float64) * valid["original_upb"]).sum() / valid[
            "original_upb"
        ].sum()
        assert 680 <= wa_fico <= 780, f"WA FICO {wa_fico:.0f} outside expected range"

    def test_wa_ltv_in_plausible_range(self, fixture_data):
        orig, _, _ = fixture_data
        valid = orig.filter(pl.col("ltv").is_not_null() & (pl.col("original_upb") > 0))
        wa_ltv = (valid["ltv"] * valid["original_upb"]).sum() / valid["original_upb"].sum()
        assert 60 <= wa_ltv <= 85, f"WA LTV {wa_ltv:.1f}% outside expected range"

    def test_prepayment_rate_plausible(self, fixture_data):
        _, perf, meta = fixture_data
        n_prepay = meta["n_prepayments"]
        n_loans = meta["n_loans"]
        rate = n_prepay / n_loans
        assert 0.05 <= rate <= 0.40, f"Prepayment rate {rate:.1%} implausible"

    def test_credit_event_rate_plausible(self, fixture_data):
        _, perf, meta = fixture_data
        n_ce = meta["n_credit_events"]
        n_loans = meta["n_loans"]
        rate = n_ce / n_loans
        assert 0.0 <= rate <= 0.15, f"Credit event rate {rate:.1%} implausible"


@pytest.mark.regression
class TestFixtureStratification:
    def test_stratification_runs_without_error(self, fixture_data):
        """Stratification engine runs on fixture data."""
        orig, perf, _ = fixture_data
        # Build a minimal panel-like DataFrame from orig
        df = orig.rename(
            {
                "loan_sequence_number": "loan_id",
                "original_upb": "current_upb",
                "original_interest_rate": "current_interest_rate",
                "credit_score": "fico",
                "ltv": "original_ltv",
                "occupancy_status": "occupancy",
            }
        )
        from stratacredit.analytics.stratification import strat_fico, strat_ltv

        fico_strat = strat_fico(df)
        ltv_strat = strat_ltv(df)
        assert len(fico_strat) > 0
        assert len(ltv_strat) > 0

    def test_stratification_pool_pct_sums_to_100(self, fixture_data):
        orig, _, _ = fixture_data
        df = orig.rename(
            {
                "loan_sequence_number": "loan_id",
                "original_upb": "current_upb",
                "original_interest_rate": "current_interest_rate",
                "credit_score": "fico",
                "ltv": "original_ltv",
                "occupancy_status": "occupancy",
            }
        )
        from stratacredit.analytics.stratification import strat_fico

        result = strat_fico(df)
        total_pct = result["pool_pct"].sum()
        assert total_pct == pytest.approx(100.0, rel=1e-3)
