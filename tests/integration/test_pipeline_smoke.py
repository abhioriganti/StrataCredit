"""Integration smoke tests: runs the full pipeline on fixture data.

These tests verify the end-to-end pipeline works without errors
using the synthetic fixture dataset.  They do NOT require real
Freddie Mac data.
"""

from pathlib import Path

import pytest


@pytest.mark.smoke
@pytest.mark.integration
class TestFixtureGeneration:
    def test_fixture_generates_parquet(self, tmp_path):
        from stratacredit.fixtures.generate import run

        result = run(output_dir=tmp_path)
        assert result["n_loans"] == 500
        assert result["n_panel_rows"] > 1000
        assert Path(result["origination_path"]).exists()
        assert Path(result["performance_path"]).exists()


@pytest.mark.smoke
class TestMetricsSmoke:
    def test_portfolio_metrics_runs(self, small_panel):
        from stratacredit.analytics.metrics import compute_portfolio_metrics

        active = small_panel.filter(small_panel["is_terminal"] == False)  # noqa
        m = compute_portfolio_metrics(active)
        assert m.loan_count > 0
        assert m.total_current_upb > 0
        assert m.wa_fico is not None

    def test_stratification_runs(self, small_panel):
        from stratacredit.analytics.stratification import strat_fico, strat_ltv

        active = small_panel.filter(small_panel["is_terminal"] == False)  # noqa
        assert len(strat_fico(active)) > 0
        assert len(strat_ltv(active)) > 0

    def test_replines_run(self, small_panel):
        from stratacredit.analytics.replines import build_replines

        active = small_panel.filter(small_panel["is_terminal"] == False)  # noqa
        replines = build_replines(active, min_cell_loans=1)
        assert len(replines) > 0
        assert "upb" in replines.columns


@pytest.mark.smoke
class TestPrepaymentSmoke:
    def test_smm_runs(self, small_panel):
        from stratacredit.performance.prepayment import compute_smm

        result = compute_smm(small_panel)
        assert len(result) >= 0  # may be empty if not enough rows
        if len(result) > 0:
            assert "smm" in result.columns
            assert "cpr" in result.columns


@pytest.mark.smoke
class TestDelinquencySmoke:
    def test_transition_matrix_runs(self, small_panel):
        from stratacredit.performance.delinquency import build_transition_matrix

        tm = build_transition_matrix(small_panel)
        assert "from_state" in tm.columns
        assert "to_state" in tm.columns

    def test_delinquency_summary_runs(self, small_panel):
        from stratacredit.performance.delinquency import delinquency_summary

        ds = delinquency_summary(small_panel)
        assert "reporting_period" in ds.columns
        assert "dq30_rate" in ds.columns


@pytest.mark.smoke
class TestCreditEventsSmoke:
    def test_credit_event_rate_runs(self, small_panel):
        from stratacredit.performance.credit_events import monthly_credit_event_rate

        result = monthly_credit_event_rate(small_panel)
        assert "cdr" in result.columns

    def test_terminations_extracted(self, small_panel):
        from stratacredit.performance.credit_events import classify_terminations

        terms = classify_terminations(small_panel)
        # B2 prepays, so at least 0 credit events (prepay is not a credit event)
        assert "loan_id" in terms.columns


@pytest.mark.smoke
class TestEligibilitySmoke:
    def test_eligibility_filter_runs(self, small_panel):
        from stratacredit.optimization.eligibility import (
            EligibilityConfig,
            apply_eligibility_filters,
        )

        cfg = EligibilityConfig(
            require_current=True,
            exclude_terminated=True,
            min_remaining_term=1,
            min_upb=1_000,
            max_upb=10_000_000,
        )
        # Use last reporting period
        snapshot = small_panel.filter(
            small_panel["reporting_period"] == small_panel["reporting_period"].max()
        )
        candidates, excl = apply_eligibility_filters(snapshot, config=cfg)
        assert "_candidate_count" in excl
        assert isinstance(candidates, type(small_panel))


@pytest.mark.smoke
class TestScenarioSmoke:
    def test_scenario_engine_runs(self, small_panel):
        from stratacredit.scenarios.engine import run_scenarios

        active = small_panel.filter(small_panel["is_terminal"] == False)  # noqa
        results = run_scenarios(
            pool=active,
            base_credit_event_rate=0.02,
            base_severity=0.35,
            base_cpr=0.18,
        )
        assert len(results) >= 3  # base + moderate + severe
        assert results[0].scenario_label is not None

    def test_base_scenario_lower_loss_than_severe(self, small_panel):
        from stratacredit.scenarios.engine import run_scenarios

        active = small_panel.filter(small_panel["is_terminal"] == False)  # noqa
        results = run_scenarios(
            pool=active,
            base_credit_event_rate=0.02,
            base_severity=0.35,
            base_cpr=0.18,
        )
        base = next(r for r in results if "Base" in r.scenario_label)
        severe = next((r for r in results if "Severe" in r.scenario_label), None)
        if severe:
            assert base.projected_gross_loss_usd <= severe.projected_gross_loss_usd
