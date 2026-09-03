"""Page 3: Historical Performance."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Historical Performance", layout="wide")
st.title("📈 Historical Performance")


@st.cache_data(ttl=300)
def load_panel():
    from stratacredit.db import get_connection

    try:
        with get_connection("silver") as conn:
            return conn.execute("""
                SELECT * FROM silver.loan_month_panel
                WHERE MOD(ABS(HASH(loan_id)), 100) = 0
            """).pl(), None
    except Exception as e:
        return None, str(e)


panel, err = load_panel()
if err or panel is None:
    st.error(f"Data not available: {err}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["CPR", "Delinquency", "Credit Events", "Loss Severity"])

with tab1:
    st.subheader("Conditional Prepayment Rate (CPR)")
    group_by = st.selectbox("Group by", ["reporting_period", "origination_year", "loan_age"])
    try:
        from stratacredit.performance.prepayment import aggregate_cpr, compute_smm

        smm_df = compute_smm(panel)
        cpr_df = aggregate_cpr(smm_df, [group_by])
        fig = px.line(
            cpr_df.to_pandas(),
            x=group_by,
            y="cpr_wa",
            title=f"CPR by {group_by}",
            labels={"cpr_wa": "CPR (%)"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cpr_df.to_pandas(), use_container_width=True)
    except Exception as e:
        st.warning(f"CPR error: {e}")

with tab2:
    st.subheader("Delinquency Rates Over Time")
    try:
        from stratacredit.performance.delinquency import (
            build_transition_matrix,
            delinquency_summary,
        )

        dq = delinquency_summary(panel)
        fig = go.Figure()
        for col, name, color in [
            ("dq30_rate", "DQ30", "#FFC107"),
            ("dq60_rate", "DQ60", "#FF9800"),
            ("dq90plus_rate", "DQ90+", "#F44336"),
            ("serious_dq_rate", "Serious DQ", "#9C27B0"),
        ]:
            if col in dq.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dq["reporting_period"].to_list(),
                        y=dq[col].to_list(),
                        name=name,
                        line={"color": color},
                    )
                )
        fig.update_layout(yaxis_title="% of Pool UPB", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transition Matrix (Latest Month)")
        tm = build_transition_matrix(panel)
        latest = tm.filter(tm["cohort_month"] == tm["cohort_month"].max()).to_pandas()
        if len(latest) > 0:
            pivot = (
                latest.pivot(index="from_state", columns="to_state", values="transition_rate_pct")
                .fillna(0)
                .round(2)
            )
            st.dataframe(pivot, use_container_width=True)
    except Exception as e:
        st.warning(f"Delinquency error: {e}")

with tab3:
    st.subheader("Credit Event Rate (CDR)")
    try:
        from stratacredit.performance.credit_events import (
            monthly_credit_event_rate,
            vintage_credit_event_matrix,
        )

        ce = monthly_credit_event_rate(panel)
        fig = px.line(
            ce.to_pandas(),
            x="reporting_period",
            y="cdr",
            title="Annualized CDR by Period",
            labels={"cdr": "CDR (%)"},
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        vce = vintage_credit_event_matrix(panel)
        fig2 = px.line(
            vce.to_pandas(),
            x="loan_age",
            y="cumulative_credit_event_rate",
            color="origination_year",
            title="Cumulative CE Rate by Vintage × Loan Age",
            labels={
                "cumulative_credit_event_rate": "Cum. CE Rate (%)",
                "loan_age": "Loan Age (mo)",
            },
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Credit event error: {e}")

with tab4:
    st.subheader("Loss Severity")
    try:
        from stratacredit.performance.severity import compute_severity, severity_summary

        sev = compute_severity(panel)
        if len(sev) > 0:
            s = severity_summary(sev)
            c1, c2, c3 = st.columns(3)
            c1.metric("Credit Events", f"{s['count']:,}")
            c2.metric("Mean Severity", f"{s['mean_severity']:.1%}")
            c3.metric("WA Severity", f"{s['wa_severity']:.1%}" if s.get("wa_severity") else "—")
            fig = px.histogram(
                sev.to_pandas(),
                x="loss_severity",
                nbins=40,
                title="Loss Severity Distribution",
                color_discrete_sequence=["#F44336"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No credit-event loans with severity data found.")
    except Exception as e:
        st.warning(f"Severity error: {e}")
