"""Unit tests for pool constraint checking."""

import polars as pl
import pytest

from stratacredit.optimization.constraints import ConstraintResult, check_pool_constraints
from stratacredit.optimization.selector import select_pool_greedy


def _make_pool(n: int = 100) -> pl.DataFrame:
    """Create a synthetic pool meeting typical constraints."""
    import random

    rng = random.Random(42)
    return pl.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "current_upb": [float(rng.randint(150_000, 500_000)) for _ in range(n)],
            "current_interest_rate": [round(rng.uniform(3.5, 5.5), 3) for _ in range(n)],
            "fico": [rng.randint(700, 800) for _ in range(n)],
            "original_ltv": [round(rng.uniform(55, 78), 1) for _ in range(n)],
            "original_cltv": [round(rng.uniform(57, 80), 1) for _ in range(n)],
            "dti": [round(rng.uniform(25, 42), 1) for _ in range(n)],
            "loan_age": [rng.randint(6, 60) for _ in range(n)],
            "remaining_term": [rng.randint(250, 354) for _ in range(n)],
            "loan_purpose": [rng.choice(["P", "R", "C"]) for _ in range(n)],
            "occupancy": [rng.choice(["P", "P", "P", "S", "I"]) for _ in range(n)],
            "property_state": [rng.choice(["CA", "TX", "FL", "NY", "IL"]) for _ in range(n)],
            "mortgage_insurance_pct": [None] * n,
            "delinquency_state": ["CURRENT"] * n,
            "modification_flag": [False] * n,
            "delinquency_months": [0] * n,
            "dq30_pct": [0.0] * n,
            "dq60_pct": [0.0] * n,
            "dq90plus_pct": [0.0] * n,
        }
    )


@pytest.mark.unit
class TestConstraintChecking:
    def test_returns_list_of_results(self):
        pool = _make_pool()
        results = check_pool_constraints(pool)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ConstraintResult) for r in results)

    def test_pass_fail_are_strings(self):
        pool = _make_pool()
        results = check_pool_constraints(pool)
        for r in results:
            assert r.status in ("PASS", "FAIL")

    def test_constraint_result_has_required_fields(self):
        pool = _make_pool()
        results = check_pool_constraints(pool)
        r = results[0]
        assert hasattr(r, "name")
        assert hasattr(r, "actual")
        assert hasattr(r, "status")

    def test_all_current_pool_passes_delinquency_constraint(self):
        pool = _make_pool()
        # All CURRENT — delinquent share should be 0%, constraint max = 0%
        results = check_pool_constraints(pool)
        dq_result = next((r for r in results if "Delinquent" in r.name), None)
        if dq_result is not None:
            assert dq_result.status == "PASS"

    def test_high_ltv_pool_fails_ltv_constraint(self):
        """A pool with 100% high-LTV loans should fail the WA LTV constraint."""
        pool = _make_pool().with_columns(pl.lit(95.0).alias("original_ltv"))
        results = check_pool_constraints(pool)
        ltv_result = next((r for r in results if "LTV" in r.name and "WA" in r.name), None)
        if ltv_result is not None:
            assert ltv_result.status == "FAIL"

    def test_selector_enforces_single_state_upb_limit(self):
        """The greedy selector must enforce a configured state concentration cap."""
        states = [state for state in ["CA", "TX", "FL", "NY", "IL"] for _ in range(30)]
        candidates = pl.DataFrame(
            {
                "loan_id": [f"L{i:03d}" for i in range(len(states))],
                "current_upb": [100_000.0] * len(states),
                "fico": [800] * len(states),
                "original_ltv": [60.0] * len(states),
                "dti": [30.0] * len(states),
                "occupancy": ["P"] * len(states),
                "property_state": states,
                "composite_score": [0.0] * len(states),
            }
        )

        selected = select_pool_greedy(
            candidates,
            target_upb=4_000_000.0,
            max_state_pct=25.0,
        )
        total_upb = float(selected["current_upb"].sum())
        by_state = selected.group_by("property_state").agg(pl.col("current_upb").sum())
        assert total_upb >= 3_980_000.0
        assert all(value / total_upb <= 0.25 for value in by_state["current_upb"].to_list())


@pytest.mark.unit
class TestStressScenario:
    """Tests for stress scenario multiplier calculations."""

    def test_base_scenario_no_change(self):
        from stratacredit.scenarios.engine import _apply_scenario

        scenario = {
            "label": "Base",
            "description": "Base",
            "default_multiplier": 1.0,
            "severity_multiplier": 1.0,
            "prepayment_multiplier": 1.0,
        }
        result = _apply_scenario(
            pool_upb=100_000_000,
            loan_count=1000,
            base_ce_rate=0.02,
            base_severity=0.35,
            base_prepay_rate=0.18,
            scenario=scenario,
        )
        assert result.stressed_credit_event_rate == pytest.approx(0.02)
        assert result.stressed_severity == pytest.approx(0.35)
        assert result.stressed_prepayment_rate == pytest.approx(0.18)

    def test_severe_stress_doubles_ce_rate(self):
        from stratacredit.scenarios.engine import _apply_scenario

        scenario = {
            "label": "Severe",
            "description": "Severe",
            "default_multiplier": 2.0,
            "severity_multiplier": 1.3,
            "prepayment_multiplier": 0.75,
        }
        result = _apply_scenario(
            pool_upb=100_000_000,
            loan_count=1000,
            base_ce_rate=0.02,
            base_severity=0.35,
            base_prepay_rate=0.18,
            scenario=scenario,
        )
        assert result.stressed_credit_event_rate == pytest.approx(0.04)
        assert result.stressed_severity == pytest.approx(0.455)
        assert result.stressed_prepayment_rate == pytest.approx(0.135)

    def test_gross_loss_calculation(self):
        from stratacredit.scenarios.engine import _apply_scenario

        scenario = {
            "label": "Test",
            "description": "Test",
            "default_multiplier": 1.0,
            "severity_multiplier": 1.0,
            "prepayment_multiplier": 1.0,
        }
        result = _apply_scenario(
            pool_upb=100_000_000,
            loan_count=1000,
            base_ce_rate=0.02,  # 2% of $100M = $2M credit events
            base_severity=0.50,  # 50% severity → $1M loss
            base_prepay_rate=0.18,
            scenario=scenario,
        )
        assert result.projected_gross_loss_usd == pytest.approx(1_000_000.0, rel=1e-3)
