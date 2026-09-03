"""Export layer: CSV, XLSX, and HTML outputs.

Generates all analyst-facing output files:
  - selected_pool.csv
  - replines.csv
  - portfolio_summary.csv
  - performance_matrices.csv
  - deal_analytics_memo.html
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUTPUTS_DIR = Path("outputs")


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def export_selected_pool(selected: pl.DataFrame, path: Path | None = None) -> Path:
    """Export selected pool to CSV."""
    out = path or ensure_outputs_dir() / "selected_pool.csv"
    selected.write_csv(out)
    return out


def export_replines(replines: pl.DataFrame, path: Path | None = None) -> Path:
    """Export replines to CSV."""
    out = path or ensure_outputs_dir() / "replines.csv"
    replines.write_csv(out)
    return out


def export_portfolio_summary(metrics, path: Path | None = None) -> Path:
    """Export portfolio summary metrics to CSV."""
    out = path or ensure_outputs_dir() / "portfolio_summary.csv"
    import dataclasses

    row = dataclasses.asdict(metrics) if dataclasses.is_dataclass(metrics) else vars(metrics)
    pl.DataFrame([row]).write_csv(out)
    return out


def export_performance_matrices(
    cpr_df: pl.DataFrame,
    dq_df: pl.DataFrame,
    path: Path | None = None,
) -> Path:
    """Export CPR history and DQ roll rates to XLSX."""
    out = path or ensure_outputs_dir() / "performance_matrices.xlsx"
    try:
        with pl.ExcelWriter(str(out)) as writer:
            cpr_df.write_excel(writer, worksheet="CPR_History")
            dq_df.write_excel(writer, worksheet="DQ_Roll_Rates")
    except Exception:
        # Fallback to CSV if openpyxl not available
        out = out.with_suffix(".csv")
        cpr_df.write_csv(out)
    return out
