"""Pool constraint definitions and pass/fail checking.

After selection, verifies every configured constraint and
produces the reconciliation table shown in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from stratacredit.analytics.metrics import compute_portfolio_metrics
from stratacredit.config import pool_config


@dataclass
class ConstraintResult:
    """One row in the constraint reconciliation table."""

    name: str
    target: float | None
    actual: float
    direction: str  # ">=" / "<=" / "~"
    unit: str  # "%" / "$" / ""
    status: str  # "PASS" / "FAIL"
    note: str = ""


def check_pool_constraints(
    selected: pl.DataFrame,
    constraints: dict | None = None,
) -> list[ConstraintResult]:
    """Check all pool constraints against the selected pool.

    Args:
        selected: Selected pool DataFrame (subset of panel snapshot).
        constraints: Override constraint dict. Loads from YAML if None.

    Returns:
        List of ConstraintResult — one per constraint checked.
    """
    cfg = constraints or _load_constraints()
    m = compute_portfolio_metrics(selected)
    results: list[ConstraintResult] = []

    tol = pool_config["reconciliation"]["float_tolerance"]

    def _pass(actual, target, direction):
        if target is None:
            return True
        if direction == ">=":
            return actual >= target - tol
        if direction == "<=":
            return actual <= target + tol
        # "~" means within tolerance band
        return abs(actual - target) <= tol

    # Pool UPB
    target_upb = cfg.get("target_upb")
    tol_pct = cfg.get("upb_tolerance_pct", 0.5) / 100.0
    if target_upb:
        lo = target_upb * (1 - tol_pct)
        hi = target_upb * (1 + tol_pct)
        ok = lo <= m.total_current_upb <= hi
        results.append(
            ConstraintResult(
                name="Pool UPB",
                target=target_upb,
                actual=m.total_current_upb,
                direction="~",
                unit="$",
                status="PASS" if ok else "FAIL",
                note=f"±{tol_pct * 100:.1f}% tolerance",
            )
        )

    # WA FICO (minimum)
    if cfg.get("min_wa_fico") and m.wa_fico is not None:
        t = cfg["min_wa_fico"]
        results.append(
            ConstraintResult(
                name="Min WA FICO",
                target=t,
                actual=m.wa_fico,
                direction=">=",
                unit="",
                status="PASS" if _pass(m.wa_fico, t, ">=") else "FAIL",
            )
        )

    # WA LTV (maximum)
    if cfg.get("max_wa_ltv") and m.wa_ltv is not None:
        t = cfg["max_wa_ltv"]
        results.append(
            ConstraintResult(
                name="Max WA LTV",
                target=t,
                actual=m.wa_ltv,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.wa_ltv, t, "<=") else "FAIL",
            )
        )

    # WA DTI (maximum)
    if cfg.get("max_wa_dti") and m.wa_dti is not None:
        t = cfg["max_wa_dti"]
        results.append(
            ConstraintResult(
                name="Max WA DTI",
                target=t,
                actual=m.wa_dti,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.wa_dti, t, "<=") else "FAIL",
            )
        )

    conc = cfg.get("concentration", {})

    # Investor share
    t = conc.get("max_investor_pct")
    if t is not None:
        results.append(
            ConstraintResult(
                name="Max Investor Share",
                target=t,
                actual=m.investor_pct,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.investor_pct, t, "<=") else "FAIL",
            )
        )

    # High-LTV share (LTV > 80)
    t = conc.get("max_high_ltv_pct")
    if t is not None:
        results.append(
            ConstraintResult(
                name="Max High-LTV Share (>80%)",
                target=t,
                actual=m.high_ltv_pct,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.high_ltv_pct, t, "<=") else "FAIL",
            )
        )

    # Low-FICO share (FICO < 660)
    t = conc.get("max_low_fico_pct")
    if t is not None:
        results.append(
            ConstraintResult(
                name="Max Low-FICO Share (<660)",
                target=t,
                actual=m.low_fico_pct,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.low_fico_pct, t, "<=") else "FAIL",
            )
        )

    # Modified share
    t = conc.get("max_modified_pct")
    if t is not None:
        results.append(
            ConstraintResult(
                name="Max Modified Share",
                target=t,
                actual=m.modified_pct,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.modified_pct, t, "<=") else "FAIL",
            )
        )

    # Delinquent share
    dq_actual = m.dq30_pct + m.dq60_pct + m.dq90plus_pct
    t = conc.get("max_delinquent_pct", 0.0)
    results.append(
        ConstraintResult(
            name="Max Delinquent Share",
            target=t,
            actual=dq_actual,
            direction="<=",
            unit="%",
            status="PASS" if _pass(dq_actual, t, "<=") else "FAIL",
        )
    )

    # Largest state concentration
    t = conc.get("max_state_pct")
    if t is not None:
        results.append(
            ConstraintResult(
                name=f"Max State Conc. ({m.largest_state})",
                target=t,
                actual=m.largest_state_pct,
                direction="<=",
                unit="%",
                status="PASS" if _pass(m.largest_state_pct, t, "<=") else "FAIL",
            )
        )

    return results


def _load_constraints() -> dict:
    """Load flat constraint dict from pool_config."""
    cfg = {}
    cfg.update(pool_config["pool"])
    cfg.update(pool_config.get("wa_constraints", {}))
    cfg["concentration"] = pool_config.get("concentration", {})
    return cfg


def constraints_to_dataframe(results: list[ConstraintResult]) -> pl.DataFrame:
    """Convert constraint results to a display DataFrame."""
    return pl.DataFrame(
        [
            {
                "Constraint": r.name,
                "Target": f"{r.unit}{r.target:,.1f}" if r.target is not None else "—",
                "Actual": f"{r.unit}{r.actual:,.1f}",
                "Direction": r.direction,
                "Status": r.status,
                "Note": r.note,
            }
            for r in results
        ]
    )
