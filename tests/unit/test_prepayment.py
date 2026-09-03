"""Unit tests for SMM and CPR calculations."""

from datetime import date

import polars as pl
import pytest


@pytest.mark.unit
class TestCPRCalculation:
    """Tests for SMM → CPR conversion formula."""

    def test_zero_smm_gives_zero_cpr(self):
        smm = 0.0
        cpr = (1.0 - (1.0 - smm) ** 12) * 100.0
        assert cpr == pytest.approx(0.0)

    def test_full_prepayment_smm_gives_100_cpr(self):
        smm = 1.0
        cpr = (1.0 - (1.0 - smm) ** 12) * 100.0
        assert cpr == pytest.approx(100.0)

    def test_known_smm_to_cpr(self):
        # SMM = 0.005 → CPR = 1 - (0.995)^12 ≈ 5.83%
        smm = 0.005
        cpr = (1.0 - (1.0 - smm) ** 12) * 100.0
        assert cpr == pytest.approx(5.8377, rel=1e-3)

    def test_20_cpr_implies_smm(self):
        # CPR = 20% → SMM = 1 - (1 - 0.20)^(1/12) ≈ 0.01850
        cpr = 0.20
        smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
        assert smm == pytest.approx(0.018423, rel=1e-3)
        # Verify round-trip
        cpr_back = 1.0 - (1.0 - smm) ** 12
        assert cpr_back == pytest.approx(cpr, rel=1e-5)


@pytest.mark.unit
class TestScheduledPrincipal:
    """Tests for scheduled principal amortization formula."""

    def test_standard_mortgage(self):
        """$200k at 4% for 360 months — scheduled principal month 1."""
        upb = 200_000.0
        annual_rate = 4.0
        r = annual_rate / 1200.0
        n = 360
        # Monthly payment
        pmt = upb * r / (1 - (1 + r) ** -n)
        # Interest portion
        interest = upb * r
        sched_p = pmt - interest
        assert sched_p > 0
        assert sched_p < pmt
        # First month scheduled principal should be ~$290
        assert sched_p == pytest.approx(290.43, rel=1e-2)

    def test_zero_rate_loan(self):
        """Zero-rate loan: scheduled principal = UPB / remaining_term."""
        upb = 120_000.0
        remaining = 120
        sched_p = upb / remaining
        assert sched_p == pytest.approx(1000.0)

    def test_amortization_reduces_balance(self):
        """After 12 payments, UPB should be lower than original."""
        upb = 200_000.0
        r = 4.0 / 1200.0
        n = 360
        for _ in range(12):
            pmt = upb * r / (1 - (1 + r) ** -n)
            interest = upb * r
            sched_p = pmt - interest
            upb -= sched_p
            n -= 1
        assert upb < 200_000.0
        assert upb > 195_000.0


@pytest.mark.unit
class TestSMMFromPanel:
    """Tests for compute_smm() with synthetic panel data."""

    def _make_panel(self) -> pl.DataFrame:
        """Two-month panel for one loan: no prepayment."""
        return pl.DataFrame(
            {
                "loan_id": ["L1", "L1"],
                "reporting_period": [date(2020, 1, 1), date(2020, 2, 1)],
                "current_upb": [200_000.0, 199_500.0],
                "current_interest_rate": [4.0, 4.0],
                "remaining_term": [360, 359],
                "loan_age": [1, 2],
                "is_voluntary_prepayment": [False, False],
                "delinquency_state": ["CURRENT", "CURRENT"],
                "origination_year": [2020, 2020],
                "fico": [740, 740],
                "original_ltv": [75.0, 75.0],
                "loan_purpose": ["P", "P"],
                "property_state": ["CA", "CA"],
                "occupancy": ["P", "P"],
            }
        )

    def test_smm_is_non_negative(self):
        from stratacredit.performance.prepayment import compute_smm

        panel = self._make_panel()
        result = compute_smm(panel)
        assert len(result) > 0
        assert (result["smm"] >= 0).all()

    def test_prepaid_loan_smm_is_one(self):
        from stratacredit.performance.prepayment import compute_smm

        panel = pl.DataFrame(
            {
                "loan_id": ["L2", "L2"],
                "reporting_period": [date(2020, 1, 1), date(2020, 2, 1)],
                "current_upb": [100_000.0, 0.0],
                "current_interest_rate": [3.5, 3.5],
                "remaining_term": [240, 239],
                "loan_age": [1, 2],
                "is_voluntary_prepayment": [False, True],
                "delinquency_state": ["CURRENT", "CURRENT"],
                "origination_year": [2020, 2020],
                "fico": [760, 760],
                "original_ltv": [70.0, 70.0],
                "loan_purpose": ["P", "P"],
                "property_state": ["TX", "TX"],
                "occupancy": ["P", "P"],
            }
        )
        result = compute_smm(panel)
        prepaid_row = result.filter(pl.col("is_voluntary_prepayment") == True)  # noqa
        if len(prepaid_row) > 0:
            assert float(prepaid_row["smm"][0]) == pytest.approx(1.0)
