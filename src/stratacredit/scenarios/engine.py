"""Stress scenario engine.

Applies configurable multipliers to base model predictions
to produce stressed credit loss, prepayment, and default projections.

These are deterministic sensitivity tools — NOT rating-agency models.
They do not replicate Fitch, Moody's, KBRA, S&P, or any proprietary
rating-agency methodology.

See docs/methodology.md and configs/pool_constraints.yaml for definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from stratacredit.config import pool_config


@dataclass
class ScenarioResult:
    """Projected outcomes under one stress scenario."""

    scenario_label: str
    description: str
    pool_upb: float
    loan_count: int

    # Multipliers applied
    default_multiplier: float
    severity_multiplier: float
    prepayment_multiplier: float

    # Base (unstressed) rates
    base_credit_event_rate: float
    base_severity: float
    base_prepayment_rate: float

    # Stressed projections
    stressed_credit_event_rate: float
    stressed_severity: float
    stressed_prepayment_rate: float

    # Projected loan counts / amounts
    projected_credit_events: int
    projected_credit_event_pct: float
    projected_prepayments: int
    projected_prepayment_pct: float

    # Loss projections
    projected_gross_loss_usd: float
    projected_gross_loss_pct: float
    projected_net_loss_usd: float  # after MI recoveries (approx)
    cumulative_loss_pct_3yr: float
    cumulative_loss_pct_5yr: float

    # Stress delta vs base
    scenario_delta_credit_events: float
    scenario_delta_loss_usd: float

    # Credit enhancement sensitivity
    indicative_loss_cushion: float | None = None  # user CE - projected loss
    ce_assumption_pct: float | None = None


def _apply_scenario(
    pool_upb: float,
    loan_count: int,
    base_ce_rate: float,  # annual credit event rate (decimal, e.g. 0.02)
    base_severity: float,  # mean loss severity (decimal, e.g. 0.35)
    base_prepay_rate: float,  # annual CPR (decimal, e.g. 0.20)
    scenario: dict,
    projection_years: int = 5,
    ce_assumption_pct: float | None = None,
) -> ScenarioResult:
    """Compute scenario projections.

    Args:
        pool_upb: Current aggregate UPB of the pool.
        loan_count: Number of loans in the pool.
        base_ce_rate: Annual credit event rate (0–1).
        base_severity: Mean loss severity (0–1.5).
        base_prepay_rate: Annual CPR (0–1).
        scenario: Scenario dict from pool_constraints.yaml.
        projection_years: Projection horizon.
        ce_assumption_pct: User-specified CE level (% of UPB) for cushion calc.

    Returns:
        ScenarioResult instance.
    """
    dm = scenario["default_multiplier"]
    sm = scenario["severity_multiplier"]
    pm = scenario["prepayment_multiplier"]

    stressed_ce_rate = base_ce_rate * dm
    stressed_severity = base_severity * sm
    stressed_prepay = base_prepay_rate * pm

    # Annual projections (simplified straight-line, no balance decline modelling)
    annual_ce_loans = int(round(loan_count * stressed_ce_rate))
    annual_ce_upb = pool_upb * stressed_ce_rate
    annual_gross_loss = annual_ce_upb * stressed_severity
    annual_prepay_loans = int(round(loan_count * stressed_prepay))

    # MI recoveries: approximate as mi_pct * ce_upb (simplified)
    # Without pool-level MI data, assume 20% of CE loans have MI at 25% coverage
    approx_mi_recovery = annual_ce_upb * 0.20 * 0.25
    annual_net_loss = max(0.0, annual_gross_loss - approx_mi_recovery)

    gross_loss_pct = stressed_ce_rate * stressed_severity * 100.0

    # Cumulative loss approximation (no prepayment correction for simplicity)
    cumulative_3yr = gross_loss_pct * min(projection_years, 3)
    cumulative_5yr = gross_loss_pct * min(projection_years, 5)

    # Stress delta vs base case
    base_gross_loss = pool_upb * base_ce_rate * base_severity
    delta_loss_usd = annual_gross_loss - base_gross_loss

    # CE cushion
    cushion = None
    if ce_assumption_pct is not None:
        ce_dollars = pool_upb * ce_assumption_pct / 100.0
        cushion = ce_dollars - annual_gross_loss

    return ScenarioResult(
        scenario_label=scenario["label"],
        description=scenario["description"],
        pool_upb=pool_upb,
        loan_count=loan_count,
        default_multiplier=dm,
        severity_multiplier=sm,
        prepayment_multiplier=pm,
        base_credit_event_rate=base_ce_rate,
        base_severity=base_severity,
        base_prepayment_rate=base_prepay_rate,
        stressed_credit_event_rate=stressed_ce_rate,
        stressed_severity=stressed_severity,
        stressed_prepayment_rate=stressed_prepay,
        projected_credit_events=annual_ce_loans,
        projected_credit_event_pct=stressed_ce_rate * 100.0,
        projected_prepayments=annual_prepay_loans,
        projected_prepayment_pct=stressed_prepay * 100.0,
        projected_gross_loss_usd=annual_gross_loss,
        projected_gross_loss_pct=gross_loss_pct,
        projected_net_loss_usd=annual_net_loss,
        cumulative_loss_pct_3yr=cumulative_3yr,
        cumulative_loss_pct_5yr=cumulative_5yr,
        scenario_delta_credit_events=annual_ce_loans - int(round(loan_count * base_ce_rate)),
        scenario_delta_loss_usd=delta_loss_usd,
        indicative_loss_cushion=cushion,
        ce_assumption_pct=ce_assumption_pct,
    )


def run_scenarios(
    pool: pl.DataFrame,
    base_credit_event_rate: float,
    base_severity: float,
    base_cpr: float,
    ce_assumption_pct: float | None = None,
    scenarios: dict | None = None,
) -> list[ScenarioResult]:
    """Run all configured stress scenarios against a pool.

    Args:
        pool: Selected pool DataFrame.
        base_credit_event_rate: Annual credit event rate from model (0–1).
        base_severity: WA loss severity from historical data (0–1).
        base_cpr: Annual CPR from model (0–1).
        ce_assumption_pct: User CE assumption (% of pool UPB) for cushion.
        scenarios: Override scenarios dict. Loads from YAML if None.

    Returns:
        List of ScenarioResult, one per scenario.
    """
    scenarios = scenarios or pool_config["scenarios"]
    pool_upb = float(pool["current_upb"].sum() or 0.0)
    loan_count = len(pool)

    results = []
    for _key, scenario in scenarios.items():
        result = _apply_scenario(
            pool_upb=pool_upb,
            loan_count=loan_count,
            base_ce_rate=base_credit_event_rate,
            base_severity=base_severity,
            base_prepay_rate=base_cpr,
            scenario=scenario,
            ce_assumption_pct=ce_assumption_pct,
        )
        results.append(result)

    return results


def scenarios_to_dataframe(results: list[ScenarioResult]) -> pl.DataFrame:
    """Convert scenario results to a display DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "Scenario": r.scenario_label,
                "Default Mult.": r.default_multiplier,
                "Severity Mult.": r.severity_multiplier,
                "Prepay Mult.": r.prepayment_multiplier,
                "Stressed CE Rate (%)": round(r.stressed_credit_event_rate * 100, 2),
                "Stressed Severity (%)": round(r.stressed_severity * 100, 2),
                "Projected CE ($M)": round(r.projected_gross_loss_usd / 1e6, 2),
                "Gross Loss (% UPB)": round(r.projected_gross_loss_pct, 3),
                "Cum. Loss 3yr (%)": round(r.cumulative_loss_pct_3yr, 3),
                "Cum. Loss 5yr (%)": round(r.cumulative_loss_pct_5yr, 3),
                "Delta vs Base ($M)": round(r.scenario_delta_loss_usd / 1e6, 2),
                "CE Cushion ($M)": round(r.indicative_loss_cushion / 1e6, 2)
                if r.indicative_loss_cushion is not None
                else None,
            }
        )
    return pl.DataFrame(rows)
