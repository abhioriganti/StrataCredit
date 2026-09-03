"""Page 2: Collateral Stratification."""

import plotly.express as px
import streamlit as st

from stratacredit.analytics.stratification import ALL_STRATIFICATIONS

st.set_page_config(page_title="Collateral Stratification", layout="wide")
st.title("🏗 Collateral Stratification")


@st.cache_data(ttl=300)
def load_panel():
    from stratacredit.db import get_connection

    try:
        with get_connection("silver") as conn:
            return conn.execute("""
                SELECT * FROM silver.loan_month_panel
                WHERE reporting_period = (
                    SELECT MAX(reporting_period) FROM silver.loan_month_panel WHERE is_terminal = FALSE
                ) AND is_terminal = FALSE
            """).pl(), None
    except Exception as e:
        return None, str(e)


df, err = load_panel()
if err or df is None:
    st.error(f"Data not available: {err}")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
purposes = ["All"] + sorted(df["loan_purpose"].drop_nulls().unique().to_list())
occ_types = ["All"] + sorted(df["occupancy"].drop_nulls().unique().to_list())
sel_purpose = st.sidebar.selectbox("Loan Purpose", purposes)
sel_occ = st.sidebar.selectbox("Occupancy", occ_types)
fico_range = st.sidebar.slider("FICO Range", 300, 850, (620, 800))
ltv_range = st.sidebar.slider("LTV Range %", 0, 110, (0, 100))

filtered = df
if sel_purpose != "All":
    filtered = filtered.filter(filtered["loan_purpose"] == sel_purpose)
if sel_occ != "All":
    filtered = filtered.filter(filtered["occupancy"] == sel_occ)
filtered = filtered.filter(
    (filtered["fico"].is_null() | filtered["fico"].is_between(fico_range[0], fico_range[1]))
    & (
        filtered["original_ltv"].is_null()
        | filtered["original_ltv"].is_between(ltv_range[0], ltv_range[1])
    )
)

st.caption(f"Showing {len(filtered):,} loans after filters")

# ── Stratification selector ───────────────────────────────────────────────────
strat_choice = st.selectbox(
    "Stratify by",
    list(ALL_STRATIFICATIONS.keys()),
    format_func=lambda x: x.replace("_", " ").title(),
)

strat_fn = ALL_STRATIFICATIONS[strat_choice]
result = strat_fn(filtered)

# Determine the band/category column name
band_col = result.columns[0]

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader(f"Stratification by {strat_choice.replace('_', ' ').title()}")
st.dataframe(
    result.to_pandas().style.format(
        {
            "current_upb": "${:,.0f}",
            "pool_pct": "{:.1f}%",
            "wa_coupon": "{:.2f}%",
            "wa_fico": "{:.0f}",
            "wa_ltv": "{:.1f}%",
            "wa_dti": "{:.1f}%",
        }
    ),
    use_container_width=True,
)

# ── Charts ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if "loan_count" in result.columns:
        fig = px.bar(
            result.to_pandas(),
            x=band_col,
            y="loan_count",
            title="Loan Count by Band",
            labels={"loan_count": "Loan Count"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "pool_pct" in result.columns:
        fig = px.bar(
            result.to_pandas(),
            x=band_col,
            y="pool_pct",
            title="Pool % by Band",
            labels={"pool_pct": "% of Pool UPB"},
            color="pool_pct",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ── WA FICO heatmap (FICO × LTV) ──────────────────────────────────────────────
st.subheader("FICO × LTV Heatmap (Loan Count)")
try:
    import polars as pl

    from stratacredit.analytics.stratification import FICO_BANDS, LTV_BANDS, _assign_band

    hm_df = filtered.with_columns(
        [
            _assign_band(filtered, "fico", FICO_BANDS).alias("fico_band"),
            _assign_band(filtered, "original_ltv", LTV_BANDS).alias("ltv_band"),
        ]
    )
    hm = (
        hm_df.group_by(["fico_band", "ltv_band"])
        .agg(pl.len().alias("count"))
        .to_pandas()
        .pivot(index="ltv_band", columns="fico_band", values="count")
        .fillna(0)
    )
    fig = px.imshow(
        hm, aspect="auto", color_continuous_scale="Blues", title="Loan Count by FICO × LTV Band"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.caption(f"Heatmap unavailable: {e}")
