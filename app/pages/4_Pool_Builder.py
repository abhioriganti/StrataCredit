"""Page 4: Pool Builder — configure constraints and run selection."""

import streamlit as st

st.set_page_config(page_title="Pool Builder", layout="wide")
st.title("🎯 Pool Builder")
st.caption("Configure collateral constraints and build a selected pool.")


@st.cache_data(ttl=300)
def load_snapshot():
    from stratacredit.db import get_connection

    try:
        with get_connection("silver") as conn:
            return conn.execute("""
                SELECT * FROM silver.loan_month_panel
                WHERE reporting_period = (
                    SELECT MAX(reporting_period)
                    FROM silver.loan_month_panel WHERE is_terminal = FALSE
                ) AND is_terminal = FALSE
            """).pl(), None
    except Exception as e:
        return None, str(e)


universe, err = load_snapshot()
if err or universe is None:
    st.error(f"Data not available: {err}")
    st.stop()

st.info(
    f"Universe: **{len(universe):,} loans** | "
    f"Total UPB: **${float(universe['current_upb'].sum()) / 1e6:.1f}M**"
)

# ── Constraint sidebar ────────────────────────────────────────────────────────
st.sidebar.header("Pool Constraints")

target_upb_m = st.sidebar.number_input("Target UPB ($M)", value=50.0, min_value=1.0, step=5.0)
tol_pct = st.sidebar.number_input("UPB Tolerance (%)", value=0.5, min_value=0.0, max_value=5.0)
min_wa_fico = st.sidebar.number_input("Min WA FICO", value=740, min_value=600, max_value=850)
max_wa_ltv = st.sidebar.number_input("Max WA LTV (%)", value=75.0, min_value=50.0, max_value=105.0)
max_wa_dti = st.sidebar.number_input("Max WA DTI (%)", value=42.0, min_value=20.0, max_value=65.0)
max_investor = st.sidebar.number_input(
    "Max Investor % (UPB)", value=10.0, min_value=0.0, max_value=50.0
)
max_state = st.sidebar.number_input(
    "Max Single-State % (UPB)", value=25.0, min_value=5.0, max_value=50.0
)
max_high_ltv = st.sidebar.number_input(
    "Max High-LTV % (>80)", value=15.0, min_value=0.0, max_value=50.0
)

run_btn = st.sidebar.button("▶  Run Pool Selection", type="primary")

if run_btn:
    from stratacredit.optimization.constraints import (
        check_pool_constraints,
        constraints_to_dataframe,
    )
    from stratacredit.optimization.eligibility import (
        EligibilityConfig,
        apply_eligibility_filters,
    )
    from stratacredit.optimization.selector import (
        compare_candidate_vs_selected,
        score_loans,
        select_pool_greedy,
    )

    with st.spinner("Running eligibility filters..."):
        cfg = EligibilityConfig(
            require_current=True,
            exclude_terminated=True,
            min_remaining_term=12,
            min_upb=10_000,
            max_upb=2_000_000,
        )
        candidates, excl = apply_eligibility_filters(universe, config=cfg)

    st.success(
        f"Candidates: **{len(candidates):,} loans** "
        f"({excl['_total_excluded']:,} excluded by eligibility)"
    )

    with st.spinner("Scoring and selecting pool..."):
        scored = score_loans(candidates)
        target_upb = target_upb_m * 1e6
        selected = select_pool_greedy(
            scored,
            target_upb,
            tol_pct,
            max_state_pct=max_state,
        )

    sel_upb = float(selected["current_upb"].sum())
    st.metric("Selected Loans", f"{len(selected):,}")
    st.metric("Selected UPB", f"${sel_upb / 1e6:.1f}M")

    # Constraint check with user overrides
    override = {
        "target_upb": target_upb,
        "upb_tolerance_pct": tol_pct,
        "min_wa_fico": min_wa_fico,
        "max_wa_ltv": max_wa_ltv,
        "max_wa_dti": max_wa_dti,
        "concentration": {
            "max_investor_pct": max_investor,
            "max_state_pct": max_state,
            "max_high_ltv_pct": max_high_ltv,
            "max_delinquent_pct": 0.0,
        },
    }
    results = check_pool_constraints(selected, constraints=override)
    passes = sum(1 for r in results if r.status == "PASS")
    fails = sum(1 for r in results if r.status == "FAIL")

    color = "🔴" if fails > 0 else "🟢"
    st.subheader(f"{color} Constraint Reconciliation — {passes} PASS / {fails} FAIL")

    cdf = constraints_to_dataframe(results).to_pandas()

    def _style(val):
        if val == "PASS":
            return "color:green;font-weight:bold"
        if val == "FAIL":
            return "color:red;font-weight:bold"
        return ""

    st.dataframe(cdf.style.applymap(_style, subset=["Status"]), use_container_width=True)

    st.subheader("Candidate vs Selected Comparison")
    comp = compare_candidate_vs_selected(candidates, selected)
    st.dataframe(comp.to_pandas(), use_container_width=True)

    # Download buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        csv = selected.write_csv()
        st.download_button(
            "⬇ Download Selected Pool (CSV)", csv, file_name="selected_pool.csv", mime="text/csv"
        )
    with col2:
        from stratacredit.analytics.replines import build_replines

        replines = build_replines(selected, min_cell_loans=1)
        rep_csv = replines.write_csv()
        st.download_button(
            "⬇ Download Replines (CSV)", rep_csv, file_name="replines.csv", mime="text/csv"
        )
else:
    st.info("Configure constraints in the sidebar and click **Run Pool Selection**.")
