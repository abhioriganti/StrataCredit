"""Greedy constrained pool selection algorithm.

Two-stage selection:
  1. Score each loan by risk/prepayment preferences
  2. Allocate target UPB across repline cells proportionally,
     then select highest-scoring loans within each cell

This is the V1 deterministic greedy approach.
See pool_selection.md for methodology and trade-offs.
"""

from __future__ import annotations

import polars as pl

from stratacredit.config import pool_config


def score_loans(
    df: pl.DataFrame,
    credit_event_score_col: str | None = None,
    prepayment_score_col: str | None = None,
    expected_loss_col: str | None = None,
) -> pl.DataFrame:
    """Compute a composite selection score for each loan.

    Lower composite_score = preferred for selection (lower risk).

    Args:
        df: Candidate pool DataFrame.
        credit_event_score_col: Column with predicted credit event probability.
        prepayment_score_col: Column with predicted prepayment probability.
        expected_loss_col: Column with expected loss amount.

    Returns:
        DataFrame with added 'composite_score' column.
    """
    prefs = pool_config["risk_preferences"]
    w_ce = prefs["weights"]["credit_event_score"]
    w_pp = prefs["weights"]["prepayment_score"]
    w_el = prefs["weights"]["expected_loss_score"]

    # Normalise available score columns to [0,1]
    exprs = []

    def _norm_col(col: str | None, alias: str) -> pl.Expr:
        if col and col in df.columns:
            c = pl.col(col)
            mn = df[col].min() or 0.0
            mx = df[col].max() or 1.0
            rng = mx - mn or 1.0
            return ((c - mn) / rng).alias(alias)
        return pl.lit(0.0).alias(alias)

    exprs = [
        _norm_col(credit_event_score_col, "_ce_norm"),
        _norm_col(prepayment_score_col, "_pp_norm"),
        _norm_col(expected_loss_col, "_el_norm"),
    ]

    df = (
        df.with_columns(exprs)
        .with_columns(
            (
                pl.col("_ce_norm") * w_ce + pl.col("_pp_norm") * w_pp + pl.col("_el_norm") * w_el
            ).alias("composite_score")
        )
        .drop(["_ce_norm", "_pp_norm", "_el_norm"])
    )

    return df


def select_pool_greedy(
    candidates: pl.DataFrame,
    target_upb: float,
    upb_tolerance_pct: float = 0.5,
    score_col: str = "composite_score",
    max_iterations: int = 10,
    max_state_pct: float | None = None,
) -> pl.DataFrame:
    """Select a pool greedily by composite score within UPB target.

    Assigns loans to repline cells, allocates UPB proportionally,
    then fills each cell by selecting loans with lowest composite_score first.

    Args:
        candidates: Scored candidate pool.
        target_upb: Target aggregate current UPB.
        upb_tolerance_pct: Allowed % deviation from target.
        score_col: Column to sort by (ascending = better).
        max_iterations: Constraint reconciliation iterations.
        max_state_pct: Maximum UPB share permitted for each property state.

    Returns:
        Selected pool DataFrame.
    """
    if score_col not in candidates.columns:
        # Fall back to random order if no score
        candidates = candidates.with_columns(pl.lit(0.0).alias(score_col))

    # First use the configured WA and concentration limits as conservative
    # loan-level guardrails. This makes the deterministic v1 selector safe:
    # a tie in risk scores cannot accidentally fill the pool with weak cells.
    cfg = pool_config
    wa = cfg.get("wa_constraints", {})
    concentration = cfg.get("concentration", {})
    eligibility = pl.lit(True)
    if wa.get("min_wa_fico") is not None and "fico" in candidates.columns:
        eligibility &= pl.col("fico") >= wa["min_wa_fico"]
    if wa.get("max_wa_ltv") is not None and "original_ltv" in candidates.columns:
        eligibility &= pl.col("original_ltv") <= wa["max_wa_ltv"]
    if wa.get("max_wa_dti") is not None and "dti" in candidates.columns:
        eligibility &= pl.col("dti") <= wa["max_wa_dti"]
    if concentration.get("max_investor_pct", 100.0) < 100.0 and "occupancy" in candidates.columns:
        eligibility &= pl.col("occupancy") != "I"

    # Global selection is deliberate: the previous cell-by-cell approach
    # allowed each cell to round up independently, materially breaching the
    # pool-size tolerance.  A stable loan-id tiebreaker makes results repeatable.
    ordered = candidates.filter(eligibility).sort([score_col, "loan_id"])
    upper_upb = target_upb * (1.0 + upb_tolerance_pct / 100.0)
    lower_upb = target_upb * (1.0 - upb_tolerance_pct / 100.0)
    configured_state_cap = concentration.get("max_state_pct")
    state_cap_pct = max_state_pct if max_state_pct is not None else configured_state_cap
    state_limit = (
        lower_upb * float(state_cap_pct) / 100.0
        if state_cap_pct is not None and "property_state" in ordered.columns
        else None
    )
    selected_ids: list[str] = []
    selected_state_upb: dict[str, float] = {}
    total_upb = 0.0
    for row in ordered.iter_rows(named=True):
        balance = float(row["current_upb"] or 0.0)
        if balance <= 0 or total_upb + balance > upper_upb:
            continue
        state = str(row.get("property_state") or "UNKNOWN")
        if state_limit is not None and selected_state_upb.get(state, 0.0) + balance > state_limit:
            continue
        selected_ids.append(row["loan_id"])
        selected_state_upb[state] = selected_state_upb.get(state, 0.0) + balance
        total_upb += balance
        if total_upb >= lower_upb:
            break

    return candidates.filter(pl.col("loan_id").is_in(selected_ids))


def compare_candidate_vs_selected(
    candidates: pl.DataFrame,
    selected: pl.DataFrame,
) -> pl.DataFrame:
    """Compare collateral metrics between candidate universe and selected pool.

    Returns a summary DataFrame with metric, candidate_value, selected_value.
    """
    from stratacredit.analytics.metrics import compute_portfolio_metrics

    cand_m = compute_portfolio_metrics(candidates)
    sel_m = compute_portfolio_metrics(selected)

    rows = []
    fields = [
        ("Total UPB ($M)", "total_current_upb", 1e6),
        ("Loan Count", "loan_count", 1.0),
        ("WA Coupon (%)", "wa_coupon", 1.0),
        ("WA FICO", "wa_fico", 1.0),
        ("WA LTV (%)", "wa_ltv", 1.0),
        ("WA DTI (%)", "wa_dti", 1.0),
        ("WALA (mo)", "wala", 1.0),
        ("Investor % (UPB)", "investor_pct", 1.0),
        ("High-LTV % (UPB)", "high_ltv_pct", 1.0),
        ("Low-FICO % (UPB)", "low_fico_pct", 1.0),
        ("Modified % (UPB)", "modified_pct", 1.0),
        ("Largest State %", "largest_state_pct", 1.0),
    ]
    for label, attr, divisor in fields:
        cv = getattr(cand_m, attr, None)
        sv = getattr(sel_m, attr, None)
        rows.append(
            {
                "Metric": label,
                "Candidate": round(cv / divisor, 2) if cv is not None else None,
                "Selected": round(sv / divisor, 2) if sv is not None else None,
                "Delta": round((sv - cv) / divisor, 2)
                if cv is not None and sv is not None
                else None,
            }
        )

    return pl.DataFrame(rows)
