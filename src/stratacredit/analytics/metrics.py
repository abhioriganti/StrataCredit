"""Portfolio collateral metrics.

Computes weighted-average and concentration metrics for any
subset of the loan_month_panel.  All weighted averages use
current UPB as the weight unless otherwise documented.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class PortfolioMetrics:
    """Collateral metrics for a portfolio snapshot."""

    as_of_date: str | None = None
    loan_count: int = 0
    total_current_upb: float = 0.0
    avg_loan_balance: float = 0.0
    wa_coupon: float | None = None
    wa_fico: float | None = None
    wa_ltv: float | None = None
    wa_cltv: float | None = None
    wa_dti: float | None = None
    wala: float | None = None
    warm: float | None = None
    purchase_pct: float = 0.0
    refinance_pct: float = 0.0
    owner_occupied_pct: float = 0.0
    investor_pct: float = 0.0
    second_home_pct: float = 0.0
    mi_share_pct: float = 0.0
    current_pct: float = 0.0
    dq30_pct: float = 0.0
    dq60_pct: float = 0.0
    dq90plus_pct: float = 0.0
    low_fico_pct: float = 0.0
    high_ltv_pct: float = 0.0
    modified_pct: float = 0.0
    largest_state: str | None = None
    largest_state_pct: float = 0.0


def weighted_average(
    df: pl.DataFrame,
    value_col: str,
    weight_col: str = "current_upb",
) -> float | None:
    """Compute UPB-weighted average of a column.

    Null values in value_col are excluded. Returns None if no valid rows.
    """
    if value_col not in df.columns or weight_col not in df.columns:
        return None
    valid = df.filter(
        pl.col(value_col).is_not_null()
        & pl.col(weight_col).is_not_null()
        & (pl.col(weight_col) > 0)
    )
    if len(valid) == 0:
        return None
    num = (valid[value_col] * valid[weight_col]).sum()
    den = valid[weight_col].sum()
    return float(num / den) if den else None


def compute_portfolio_metrics(df: pl.DataFrame) -> PortfolioMetrics:
    """Compute full portfolio metrics from a panel snapshot.

    Args:
        df: Snapshot of loan_month_panel rows (one reporting period,
            active loans only).

    Returns:
        PortfolioMetrics instance.
    """
    if len(df) == 0:
        return PortfolioMetrics()

    total_upb = float(df["current_upb"].sum() or 0.0)
    n = len(df)

    def pct(mask: pl.Series) -> float:
        if total_upb == 0:
            return 0.0
        return float((df["current_upb"].filter(mask).sum() or 0.0) / total_upb * 100)

    # Purpose
    purchase_pct = pct(df["loan_purpose"] == "P") if "loan_purpose" in df.columns else 0.0
    refi_pct = pct(df["loan_purpose"].is_in(["C", "R"])) if "loan_purpose" in df.columns else 0.0

    # Occupancy
    occ_col = "occupancy" if "occupancy" in df.columns else None
    occ_pct = pct(df[occ_col] == "P") if occ_col else 0.0
    inv_pct = pct(df[occ_col] == "I") if occ_col else 0.0
    s2_pct = pct(df[occ_col] == "S") if occ_col else 0.0

    # MI
    mi_share = 0.0
    if "mortgage_insurance_pct" in df.columns:
        mi_share = pct(
            (df["mortgage_insurance_pct"].cast(pl.Float64, strict=False) > 0).fill_null(False)
        )

    # Delinquency
    dq_state = df["delinquency_state"] if "delinquency_state" in df.columns else None
    cur_pct = pct(dq_state == "CURRENT") if dq_state is not None else 0.0
    d30_pct = pct(dq_state == "DQ30") if dq_state is not None else 0.0
    d60_pct = pct(dq_state == "DQ60") if dq_state is not None else 0.0
    d90_pct = pct(dq_state.is_in(["DQ90PLUS", "REO"])) if dq_state is not None else 0.0

    # Concentrations
    low_fico = pct((df["fico"] < 660).fill_null(False)) if "fico" in df.columns else 0.0
    high_ltv = (
        pct((df["original_ltv"] > 80).fill_null(False)) if "original_ltv" in df.columns else 0.0
    )
    modified = (
        pct(df["modification_flag"].fill_null(False)) if "modification_flag" in df.columns else 0.0
    )

    # Largest state
    largest_state, largest_state_pct = None, 0.0
    if "property_state" in df.columns and total_upb > 0:
        state_upb = (
            df.group_by("property_state")
            .agg(pl.col("current_upb").sum().alias("su"))
            .sort("su", descending=True)
        )
        if len(state_upb) > 0:
            top = state_upb.row(0, named=True)
            largest_state = top["property_state"]
            largest_state_pct = float(top["su"] / total_upb * 100)

    return PortfolioMetrics(
        loan_count=n,
        total_current_upb=total_upb,
        avg_loan_balance=total_upb / n if n > 0 else 0.0,
        wa_coupon=weighted_average(df, "current_interest_rate"),
        wa_fico=weighted_average(df, "fico"),
        wa_ltv=weighted_average(df, "original_ltv"),
        wa_cltv=weighted_average(df, "original_cltv"),
        wa_dti=weighted_average(df, "dti"),
        wala=weighted_average(df, "loan_age"),
        warm=weighted_average(df, "remaining_term"),
        purchase_pct=purchase_pct,
        refinance_pct=refi_pct,
        owner_occupied_pct=occ_pct,
        investor_pct=inv_pct,
        second_home_pct=s2_pct,
        mi_share_pct=mi_share,
        current_pct=cur_pct,
        dq30_pct=d30_pct,
        dq60_pct=d60_pct,
        dq90plus_pct=d90_pct,
        low_fico_pct=low_fico,
        high_ltv_pct=high_ltv,
        modified_pct=modified,
        largest_state=largest_state,
        largest_state_pct=largest_state_pct,
    )
