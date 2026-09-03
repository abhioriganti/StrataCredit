"""StrataCredit Streamlit Analyst Workbench.

Launch with:  streamlit run app/streamlit_app.py
              make app
"""

import streamlit as st

st.set_page_config(
    page_title="StrataCredit Analyst Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("StrataCredit")
st.sidebar.caption("Loan-Level Structured Credit Analytics")
st.sidebar.divider()

st.title("StrataCredit Analyst Workbench")
st.markdown("""
Welcome to the **StrataCredit** analyst workbench — a structured-finance analytics
platform for mortgage collateral analysis, pool selection, and stress testing.

### Navigation

Use the **Pages** menu in the sidebar to navigate:

| Page | Description |
|------|-------------|
| 📋 Executive Snapshot | Pool UPB, WA metrics, concentrations, constraint status |
| 🏗 Collateral Stratification | FICO/LTV/DTI/coupon/vintage breakdowns |
| 📈 Historical Performance | CPR curves, delinquency transitions, credit events |
| 🎯 Pool Builder | Set constraints, run selection, download tape |
| 🤖 Model & Backtest | OOT validation, ROC/PR, feature importance |
| 🔍 Data Quality | Validation results, quarantine records, reconciliation |

### Status
""")

# Show data availability
col1, col2, col3 = st.columns(3)

with col1:
    try:
        from stratacredit.db import get_connection

        with get_connection("silver") as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM silver.loan_month_panel WHERE is_terminal = FALSE"
            ).fetchone()[0]
        st.metric("Active Loan-Month Rows", f"{n:,}")
    except Exception:
        st.metric("Active Loan-Month Rows", "Not loaded")
        st.caption("Run `make pipeline` to ingest data.")

with col2:
    try:
        from stratacredit.db import get_connection

        with get_connection("silver") as conn:
            n = conn.execute(
                "SELECT COUNT(DISTINCT loan_id) FROM silver.loan_origination"
            ).fetchone()[0]
        st.metric("Unique Loans", f"{n:,}")
    except Exception:
        st.metric("Unique Loans", "—")

with col3:
    try:
        from stratacredit.db import get_connection

        with get_connection("silver") as conn:
            row = conn.execute(
                "SELECT MIN(origination_year), MAX(origination_year) FROM silver.loan_month_panel"
            ).fetchone()
        if row and row[0]:
            st.metric("Vintage Range", f"{row[0]}–{row[1]}")
        else:
            st.metric("Vintage Range", "—")
    except Exception:
        st.metric("Vintage Range", "—")

st.divider()
st.caption(
    "StrataCredit v0.1.0 | Based on Freddie Mac Single-Family Loan-Level Dataset | "
    "For research and informational purposes only. Not investment advice."
)
