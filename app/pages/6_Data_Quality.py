"""Page 6: Data Quality — validation results, quarantine, reconciliation."""

import streamlit as st

st.set_page_config(page_title="Data Quality", layout="wide")
st.title("🔍 Data Quality")
st.caption("Validation results, quarantine records, and reconciliation checks.")


@st.cache_data(ttl=120)
def run_dq():
    from stratacredit.db import get_connection
    from stratacredit.quality.reconciliation import run_all_reconciliations
    from stratacredit.quality.rules import ORIGINATION_RULES, PERFORMANCE_RULES
    from stratacredit.quality.validator import results_to_dataframe, run_validation

    out = {"orig": None, "perf": None, "recon": None, "error": None}
    try:
        with get_connection("silver") as conn:
            orig_df = conn.execute("SELECT * FROM silver.loan_origination").pl()
            perf_df = conn.execute("SELECT * FROM silver.loan_performance LIMIT 200000").pl()

        orig_results, _ = run_validation(orig_df, ORIGINATION_RULES)
        perf_results, _ = run_validation(perf_df, PERFORMANCE_RULES)
        recon = run_all_reconciliations()

        out["orig"] = results_to_dataframe(orig_results).to_pandas()
        out["perf"] = results_to_dataframe(perf_results).to_pandas()
        out["recon"] = recon
    except Exception as e:
        out["error"] = str(e)
    return out


if st.button("🔄 Run Data Quality Checks"):
    st.cache_data.clear()

data = run_dq()

if data["error"]:
    st.error(f"DQ check failed: {data['error']}")
    st.info("Run `make pipeline` first to build Silver tables.")
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
all_results = []
if data["orig"] is not None:
    all_results.extend(data["orig"].to_dict("records"))
if data["perf"] is not None:
    all_results.extend(data["perf"].to_dict("records"))

total = len(all_results)
passes = sum(1 for r in all_results if r.get("status") == "PASS")
warns = sum(1 for r in all_results if r.get("status") == "WARN")
fails = sum(1 for r in all_results if r.get("status") == "FAIL")
quarantined = sum(1 for r in all_results if r.get("status") == "QUARANTINE")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rules", total)
c2.metric("Pass", passes, delta=None)
c3.metric("Warn", warns)
c4.metric(
    "Fail / Quarantine",
    f"{fails} / {quarantined}",
    delta=f"{fails + quarantined} issues" if fails + quarantined > 0 else "Clean",
    delta_color="inverse",
)

st.divider()

# ── Origination rules ─────────────────────────────────────────────────────────
if data["orig"] is not None:
    st.subheader("Origination Validation")

    def _style(val):
        m = {
            "PASS": "color:green",
            "FAIL": "color:red;font-weight:bold",
            "WARN": "color:orange",
            "QUARANTINE": "color:darkorange;font-weight:bold",
        }
        return m.get(val, "")

    st.dataframe(
        data["orig"].style.applymap(_style, subset=["status"]),
        use_container_width=True,
    )

# ── Performance rules ─────────────────────────────────────────────────────────
if data["perf"] is not None:
    st.subheader("Performance Validation (sample 200k rows)")
    st.dataframe(
        data["perf"].style.applymap(_style, subset=["status"]),
        use_container_width=True,
    )

# ── Reconciliation ────────────────────────────────────────────────────────────
if data["recon"]:
    st.subheader("Source-to-Target Reconciliation")
    import pandas as pd

    recon_rows = [
        {
            "Check": r.check_name,
            "Source": f"{r.source_count:,}",
            "Target": f"{r.target_count:,}",
            "Difference": f"{r.difference:,}",
            "Diff %": f"{r.pct_difference:.2f}%",
            "Status": r.status,
            "Note": r.note,
        }
        for r in data["recon"]
    ]
    df = pd.DataFrame(recon_rows)
    st.dataframe(
        df.style.applymap(_style, subset=["Status"]),
        use_container_width=True,
    )
