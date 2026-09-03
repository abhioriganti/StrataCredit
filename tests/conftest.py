"""Shared pytest fixtures for StrataCredit test suite."""

from datetime import date

import polars as pl
import pytest


@pytest.fixture(scope="session")
def small_panel() -> pl.DataFrame:
    """A minimal loan_month_panel with 3 loans × 3 months each."""
    rows = []
    loans = [
        {
            "loan_id": "A1",
            "fico": 740,
            "original_ltv": 75.0,
            "dti": 35.0,
            "original_interest_rate": 4.0,
            "original_upb": 200_000.0,
            "occupancy": "P",
            "loan_purpose": "P",
            "property_state": "CA",
            "origination_year": 2020,
        },
        {
            "loan_id": "B2",
            "fico": 680,
            "original_ltv": 82.0,
            "dti": 42.0,
            "original_interest_rate": 4.5,
            "original_upb": 150_000.0,
            "occupancy": "I",
            "loan_purpose": "C",
            "property_state": "TX",
            "origination_year": 2019,
        },
        {
            "loan_id": "C3",
            "fico": 780,
            "original_ltv": 65.0,
            "dti": 28.0,
            "original_interest_rate": 3.5,
            "original_upb": 300_000.0,
            "occupancy": "P",
            "loan_purpose": "P",
            "property_state": "FL",
            "origination_year": 2021,
        },
    ]
    periods = [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)]
    upbs = {
        "A1": [200_000.0, 199_700.0, 199_400.0],
        "B2": [150_000.0, 149_800.0, 0.0],  # prepays month 3
        "C3": [300_000.0, 299_600.0, 299_200.0],
    }
    for loan in loans:
        for i, period in enumerate(periods):
            upb = upbs[loan["loan_id"]][i]
            is_terminal = loan["loan_id"] == "B2" and i == 2
            is_prepay = is_terminal
            rows.append(
                {
                    **loan,
                    "reporting_period": period,
                    "current_upb": upb,
                    "current_interest_rate": loan["original_interest_rate"],
                    "loan_age": i + 1,
                    "remaining_term": 360 - i - 1,
                    "delinquency_months": 0,
                    "delinquency_state": "TERMINAL" if is_terminal else "CURRENT",
                    "modification_flag": False,
                    "is_terminal": is_terminal,
                    "is_voluntary_prepayment": is_prepay,
                    "is_credit_event": False,
                    "is_seriously_delinquent": False,
                    "zero_balance_code": "01" if is_prepay else None,
                    "zero_balance_date": period if is_prepay else None,
                    "zero_balance_removal_upb": 149_800.0 if is_prepay else None,
                    "actual_loss": None,
                    "mortgage_insurance_pct": None,
                    "original_cltv": loan["original_ltv"] + 2.0,
                    "original_loan_term": 360,
                    "reporting_year": period.year,
                    "reporting_month": period.month,
                }
            )
    return pl.DataFrame(rows)
