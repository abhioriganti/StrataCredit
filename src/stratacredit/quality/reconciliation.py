"""Source-to-target reconciliation checks.

Verifies row counts and UPB totals reconcile across data layers:
  Bronze -> Silver -> Gold
"""

from __future__ import annotations

from dataclasses import dataclass

from stratacredit.db import get_connection


@dataclass
class ReconciliationResult:
    check_name: str
    source_count: int
    target_count: int
    difference: int
    pct_difference: float
    status: str  # PASS / WARN / FAIL
    note: str = ""


def _pct_diff(source: int, target: int) -> float:
    if source == 0:
        return 0.0
    return abs(source - target) / source * 100.0


def reconcile_silver_counts() -> list[ReconciliationResult]:
    """Check that origination and performance loan_id sets are consistent."""
    results = []
    with get_connection("silver") as conn:
        orig_count = conn.execute(
            "SELECT COUNT(DISTINCT loan_id) FROM silver.loan_origination"
        ).fetchone()[0]
        perf_count = conn.execute(
            "SELECT COUNT(DISTINCT loan_id) FROM silver.loan_performance"
        ).fetchone()[0]
        panel_count = conn.execute(
            "SELECT COUNT(DISTINCT loan_id) FROM silver.loan_month_panel"
        ).fetchone()[0]

        # Panel should equal origination (inner join loses orphan perf records)
        diff = orig_count - panel_count
        pct = _pct_diff(orig_count, panel_count)
        results.append(
            ReconciliationResult(
                check_name="origination_vs_panel_unique_loans",
                source_count=orig_count,
                target_count=panel_count,
                difference=diff,
                pct_difference=pct,
                status="PASS" if pct < 1.0 else "WARN",
                note="Loans in origination with no performance data are excluded from panel",
            )
        )

        # Performance loan IDs not in origination
        orphan_count = conn.execute("""
            SELECT COUNT(DISTINCT p.loan_id)
            FROM silver.loan_performance p
            LEFT JOIN silver.loan_origination o ON p.loan_id = o.loan_id
            WHERE o.loan_id IS NULL
        """).fetchone()[0]
        results.append(
            ReconciliationResult(
                check_name="performance_orphan_loan_ids",
                source_count=perf_count,
                target_count=orphan_count,
                difference=orphan_count,
                pct_difference=_pct_diff(perf_count, perf_count - orphan_count),
                status="PASS" if orphan_count == 0 else "WARN",
                note="Performance records with no origination match",
            )
        )

    return results


def reconcile_upb() -> list[ReconciliationResult]:
    """Check that UPB totals reconcile between performance and panel."""
    results = []
    with get_connection("silver") as conn:
        perf_upb = conn.execute(
            "SELECT COALESCE(SUM(current_upb), 0) FROM silver.loan_performance"
        ).fetchone()[0]
        panel_upb = conn.execute(
            "SELECT COALESCE(SUM(current_upb), 0) FROM silver.loan_month_panel"
        ).fetchone()[0]

        pct = _pct_diff(int(perf_upb), int(panel_upb))
        results.append(
            ReconciliationResult(
                check_name="performance_vs_panel_total_upb",
                source_count=int(perf_upb),
                target_count=int(panel_upb),
                difference=int(perf_upb - panel_upb),
                pct_difference=pct,
                status="PASS" if pct < 0.1 else "WARN",
                note="Total current UPB should match between performance and panel",
            )
        )
    return results


def run_all_reconciliations() -> list[ReconciliationResult]:
    """Run all reconciliation checks."""
    results = []
    for fn in (reconcile_silver_counts, reconcile_upb):
        try:
            results.extend(fn())
        except Exception as e:
            results.append(
                ReconciliationResult(
                    check_name=fn.__name__,
                    source_count=0,
                    target_count=0,
                    difference=0,
                    pct_difference=0.0,
                    status="FAIL",
                    note=str(e),
                )
            )
    return results
