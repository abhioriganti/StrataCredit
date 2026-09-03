"""Page 1: Executive Deal Snapshot."""

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Executive Snapshot", layout="wide")
st.title("📋 Executive Deal Snapshot")

# ── Load data ─────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_snapshot():
    from stratacredit.analytics.metrics import compute_portfolio_metrics
    from stratacredit.db import get_connection

    try:
        with get_connection("silver") as conn:
            df = conn.execute("""
                SELECT * FROM silver.loan_month_panel
                WHERE reporting_period = (
                    SELECT MAX(reporting_period)
                    FROM silver.loan_month_panel
                    WHERE is_terminal = FALSE
                )
                AND is_terminal = FALSE
            """).pl()
        return df, compute_portfolio_metrics(df), None
    except Exception as e:
        return None, None, str(e)


df, m, err = load_snapshot()

if err:
    st.error(f"Data not available: {err}")
    st.info("Run `make pipeline` to ingest and transform data, then refresh.")
    st.stop()

st.caption(
    "Current active collateral universe at the latest available reporting period. "
    "Build a selected pool on the Pool Builder page for transaction-specific results."
)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Active Universe UPB", f"${m.total_current_upb / 1e6:.1f}M")
c2.metric("Loans", f"{m.loan_count:,}")
c3.metric("WAC", f"{m.wa_coupon:.2f}%" if m.wa_coupon else "—")
c4.metric("WA FICO", f"{m.wa_fico:.0f}" if m.wa_fico else "—")
c5.metric("WA LTV", f"{m.wa_ltv:.1f}%" if m.wa_ltv else "—")
c6.metric("WALA", f"{m.wala:.0f} mo" if m.wala else "—")

c1b, c2b, c3b, c4b = st.columns(4)
c1b.metric("WA DTI", f"{m.wa_dti:.1f}%" if m.wa_dti else "—")
c2b.metric("Purchase %", f"{m.purchase_pct:.1f}%")
c3b.metric("Investor %", f"{m.investor_pct:.1f}%")
c4b.metric("MI Share %", f"{m.mi_share_pct:.1f}%")

st.divider()

# ── Delinquency & concentrations ──────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Delinquency Breakdown")
    fig = go.Figure(
        go.Bar(
            x=["Current", "DQ30", "DQ60", "DQ90+"],
            y=[m.current_pct, m.dq30_pct, m.dq60_pct, m.dq90plus_pct],
            marker_color=["#2196F3", "#FFC107", "#FF9800", "#F44336"],
        )
    )
    fig.update_layout(yaxis_title="% of Active Universe UPB", height=300, margin={"t": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Key Concentrations")
    conc_data = {
        "Low FICO (<660)": m.low_fico_pct,
        "High LTV (>80%)": m.high_ltv_pct,
        "Modified": m.modified_pct,
        f"Top State ({m.largest_state})": m.largest_state_pct,
    }
    for label, val in conc_data.items():
        flag = "🔴" if val > 15 else "🟡" if val > 8 else "🟢"
        st.metric(label, f"{val:.1f}%", delta=None)

# ── Constraint summary (if pool has been selected) ────────────────────────────
st.divider()
st.subheader("Reference Constraint Comparison")
st.caption(
    "These checks apply the configured pool constraints to the entire active universe; they are not selected-pool results."
)
try:
    from stratacredit.optimization.constraints import (
        check_pool_constraints,
        constraints_to_dataframe,
    )

    results = check_pool_constraints(df)
    cdf = constraints_to_dataframe(results)
    passes = sum(1 for r in results if r.status == "PASS")
    fails = sum(1 for r in results if r.status == "FAIL")
    st.metric(
        "Universe vs constraints",
        f"{passes} PASS / {fails} FAIL",
        delta=f"{fails} issues" if fails else "All clear",
        delta_color="inverse",
    )

    def _color_status(val):
        if val == "PASS":
            return "color: green; font-weight: bold"
        if val == "FAIL":
            return "color: red; font-weight: bold"
        return ""

    st.dataframe(
        cdf.to_pandas().style.applymap(_color_status, subset=["Status"]),
        use_container_width=True,
    )
except Exception:
    st.info(
        "Configure and run a selection on the Pool Builder page to generate selected-pool constraint results."
    )
