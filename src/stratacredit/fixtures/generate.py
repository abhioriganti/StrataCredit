"""Synthetic test fixture generator.

Generates a small, deterministic portfolio of synthetic mortgage loans
with known expected outputs.  Used for CI/CD and regression testing.

Fixed seed = 42 ensures reproducibility.

Output:
    data/fixtures/origination.parquet
    data/fixtures/performance.parquet

Stats (deterministic):
    ~500 unique loans
    ~18,000 loan-month observations (36 months average)
    Vintages: 2018–2022
    Voluntary prepayments: ~15% of loans
    Credit events: ~3% of loans
    Serious delinquencies: ~5% of loans
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import polars as pl

from stratacredit import DATA_DIR

FIXTURES_DIR = DATA_DIR / "fixtures"
SEED = 42
N_LOANS = 500
MAX_MONTHS = 48
STATES = [
    "CA",
    "TX",
    "FL",
    "NY",
    "IL",
    "PA",
    "OH",
    "GA",
    "NC",
    "WA",
    "AZ",
    "CO",
    "VA",
    "MA",
    "MD",
    "NJ",
    "MN",
    "WI",
    "MO",
    "TN",
]
CHANNELS = ["R", "B", "C"]
LOAN_PURPOSES = ["P", "C", "R"]
OCCUPANCIES = ["P", "P", "P", "S", "I"]  # weighted toward primary
PROPERTY_TYPES = ["SF", "SF", "SF", "CO", "PU"]
VINTAGES = [2018, 2019, 2020, 2021, 2022]


def _rand_fico(rng: random.Random) -> int:
    return int(rng.gauss(730, 50))


def _rand_ltv(rng: random.Random) -> float:
    return round(rng.gauss(72, 12), 1)


def _rand_dti(rng: random.Random) -> float:
    return round(rng.gauss(36, 8), 1)


def _rand_rate(rng: random.Random, vintage: int) -> float:
    # Rates decline from 2018 to 2021, rise in 2022
    base = {2018: 4.5, 2019: 4.0, 2020: 3.2, 2021: 3.0, 2022: 5.5}[vintage]
    return round(base + rng.gauss(0, 0.3), 3)


def _rand_upb(rng: random.Random) -> float:
    return round(rng.uniform(80_000, 600_000), -2)


def generate_origination(rng: random.Random) -> list[dict]:
    rows = []
    for i in range(N_LOANS):
        vintage = rng.choice(VINTAGES)
        # first_payment_date: random month in vintage year
        month = rng.randint(1, 12)
        fpd = f"{month:02d}{vintage}"
        # maturity = 30 years later
        mat_year = vintage + 30
        mat_month = month
        maturity = f"{mat_month:02d}{mat_year}"

        fico = max(300, min(850, _rand_fico(rng)))
        ltv = max(20.0, min(105.0, _rand_ltv(rng)))
        cltv = min(ltv + rng.uniform(0, 5), 110.0)
        dti = max(5.0, min(65.0, _rand_dti(rng)))
        rate = _rand_rate(rng, vintage)
        upb = _rand_upb(rng)

        rows.append(
            {
                "loan_sequence_number": f"F{vintage}{i:05d}",
                "credit_score": fico,
                "first_payment_date": fpd,
                "first_time_homebuyer_flag": "Y" if rng.random() < 0.15 else "N",
                "maturity_date": maturity,
                "msa": str(rng.randint(10000, 49999)),
                "mi_pct": round(rng.uniform(0.3, 1.5), 2) if ltv > 80 else None,
                "num_units": 1,
                "occupancy_status": rng.choice(OCCUPANCIES),
                "cltv": round(cltv, 1),
                "dti": dti,
                "original_upb": upb,
                "ltv": ltv,
                "original_interest_rate": rate,
                "channel": rng.choice(CHANNELS),
                "prepayment_penalty_flag": "N",
                "amort_type": "FRM",
                "property_state": rng.choice(STATES),
                "property_type": rng.choice(PROPERTY_TYPES),
                "postal_code": f"{rng.randint(10000, 99999):05d}",
                "loan_purpose": rng.choice(LOAN_PURPOSES),
                "original_loan_term": 360,
                "num_borrowers": rng.choice([1, 2]),
                "seller_name": "FIXTURE BANK",
                "servicer_name": "FIXTURE SERVICER",
                "super_conforming_flag": "N",
                "pre_harp_sequence_number": None,
                "program_indicator": None,
                "harp_indicator": "N",
                "property_valuation_method": "1",
                "interest_only_indicator": "N",
                "mi_cancellation_indicator": "N",
                "_source_file": "fixture_orig.txt",
                "_source_vintage": str(vintage),
                "_ingested_at": "2024-01-01T00:00:00+00:00",
                "_file_checksum": "fixture_checksum",
            }
        )
    return rows


def generate_performance(orig_rows: list[dict], rng: random.Random) -> list[dict]:
    rows = []
    for o in orig_rows:
        loan_id = o["loan_sequence_number"]
        vintage = int(o["_source_vintage"])
        # Parse first payment date
        fpd_raw = o["first_payment_date"]
        fpd_month = int(fpd_raw[:2])
        fpd_year = int(fpd_raw[2:])
        fpd = date(fpd_year, fpd_month, 1)

        upb = float(o["original_upb"])
        rate = float(o["original_interest_rate"])
        term = int(o["original_loan_term"])
        monthly_rate = rate / 1200.0

        # Loan fate
        fate = rng.random()
        if fate < 0.03:
            fate_type = "credit_event"
            fate_month = rng.randint(12, min(MAX_MONTHS, 36))
        elif fate < 0.18:
            fate_type = "prepay"
            fate_month = rng.randint(6, MAX_MONTHS)
        else:
            fate_type = "active"
            fate_month = MAX_MONTHS

        # DQ path for credit events
        dq_start = fate_month - rng.randint(3, 6) if fate_type == "credit_event" else None

        for age in range(1, fate_month + 1):
            period = date(fpd.year + (fpd.month + age - 2) // 12, (fpd.month + age - 2) % 12 + 1, 1)
            rp = f"{period.month:02d}{period.year}"
            remaining = term - age

            # Scheduled principal
            if monthly_rate > 0:
                sched_p = upb * monthly_rate / ((1 + monthly_rate) ** remaining - 1)
            else:
                sched_p = upb / remaining if remaining > 0 else 0

            # Delinquency
            if dq_start and age >= dq_start:
                dq_months = age - dq_start
                dq_status = str(min(dq_months + 1, 9))
            else:
                dq_status = "0"

            # Terminal event
            zbc = None
            zb_date = None
            zb_upb = None
            actual_loss = None

            if fate_type == "prepay" and age == fate_month:
                zbc = "01"
                zb_date = rp
                zb_upb = round(upb, 2)
                upb = 0.0
            elif fate_type == "credit_event" and age == fate_month:
                zbc = rng.choice(["02", "03", "09"])
                zb_date = rp
                zb_upb = round(upb, 2)
                severity = rng.uniform(0.10, 0.60)
                actual_loss = round(upb * severity, 2)
                upb = 0.0
            else:
                upb = max(0.0, round(upb - sched_p, 2))

            rows.append(
                {
                    "loan_sequence_number": loan_id,
                    "monthly_reporting_period": rp,
                    "current_actual_upb": upb if zbc is None else 0.0,
                    "current_loan_delinquency_status": dq_status,
                    "loan_age": age,
                    "remaining_months_to_legal_maturity": max(0, remaining),
                    "defect_settlement_date": None,
                    "modification_flag": "N",
                    "zero_balance_code": zbc,
                    "zero_balance_effective_date": zb_date,
                    "current_interest_rate": rate,
                    "current_deferred_upb": None,
                    "ddlpi": None,
                    "mi_recoveries": round(upb * 0.05, 2)
                    if zbc in ("02", "03", "09") and o["mi_pct"]
                    else None,
                    "net_sale_proceeds": str(round(upb * rng.uniform(0.5, 0.9), 2))
                    if zbc in ("02", "03", "09")
                    else None,
                    "non_mi_recoveries": None,
                    "expenses": round(upb * 0.03, 2) if zbc in ("02", "03", "09") else None,
                    "legal_costs": None,
                    "maintenance_costs": None,
                    "taxes_insurance": None,
                    "miscellaneous_expenses": None,
                    "actual_loss_calculation": actual_loss,
                    "modification_cost": None,
                    "step_modification_flag": "N",
                    "deferred_payment_plan": "N",
                    "estimated_ltv": round(
                        float(o["ltv"]) * (upb / float(o["original_upb"]))
                        if float(o["original_upb"]) > 0
                        else float(o["ltv"]),
                        1,
                    )
                    if upb > 0
                    else None,
                    "zero_balance_removal_upb": zb_upb,
                    "delinquent_accrued_interest": None,
                    "delinquency_due_to_disaster": "N",
                    "borrower_assistance_status_code": None,
                    "current_month_modification_cost": None,
                    "interest_bearing_upb": upb if zbc is None else None,
                    "_source_file": "fixture_perf.txt",
                    "_source_vintage": str(vintage),
                    "_ingested_at": "2024-01-01T00:00:00+00:00",
                    "_file_checksum": "fixture_checksum",
                }
            )

            if zbc is not None:
                break  # no more rows after terminal event

    return rows


def run(output_dir: Path | None = None) -> dict:
    """Generate fixture data and write Parquet files.

    Returns:
        Dict with paths and counts.
    """
    out_dir = output_dir or FIXTURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    print("Generating origination fixtures...")
    orig_rows = generate_origination(rng)
    orig_df = pl.DataFrame(orig_rows)
    orig_path = out_dir / "origination.parquet"
    orig_df.write_parquet(orig_path)
    print(f"  {len(orig_df):,} origination rows → {orig_path}")

    print("Generating performance fixtures...")
    perf_rows = generate_performance(orig_rows, rng)
    perf_df = pl.DataFrame(perf_rows, infer_schema_length=None)
    perf_path = out_dir / "performance.parquet"
    perf_df.write_parquet(perf_path)
    print(f"  {len(perf_df):,} performance rows → {perf_path}")

    n_prepay = perf_df.filter(pl.col("zero_balance_code") == "01").shape[0]
    n_ce = perf_df.filter(pl.col("zero_balance_code").is_in(["02", "03", "09"])).shape[0]
    print("\nFixture summary:")
    print(f"  Unique loans:    {len(orig_df):,}")
    print(f"  Panel rows:      {len(perf_df):,}")
    print(f"  Prepayments:     {n_prepay:,}")
    print(f"  Credit events:   {n_ce:,}")

    return {
        "origination_path": orig_path,
        "performance_path": perf_path,
        "n_loans": len(orig_df),
        "n_panel_rows": len(perf_df),
        "n_prepayments": n_prepay,
        "n_credit_events": n_ce,
    }


if __name__ == "__main__":
    run()
