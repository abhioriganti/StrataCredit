"""Utility functions for working with the loan_month_panel.

Provides helpers for common panel queries used throughout
the analytics and modeling modules.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from stratacredit.db import get_connection


def load_panel_as_of(
    cutoff_date: date,
    active_only: bool = True,
) -> pl.DataFrame:
    """Load panel rows for the reporting period closest to cutoff_date.

    Args:
        cutoff_date: The as-of date for the snapshot.
        active_only: If True, exclude terminated loans.

    Returns:
        Polars DataFrame of panel rows.
    """
    where = "AND is_terminal = FALSE" if active_only else ""
    sql = f"""
        SELECT *
        FROM silver.loan_month_panel
        WHERE reporting_period = (
            SELECT MAX(reporting_period)
            FROM silver.loan_month_panel
            WHERE reporting_period <= CAST('{cutoff_date}' AS DATE)
            {where}
        )
        {where}
    """
    with get_connection("silver") as conn:
        return conn.execute(sql).pl()


def load_panel_range(
    start_date: date,
    end_date: date,
    active_only: bool = False,
) -> pl.DataFrame:
    """Load all panel rows within a reporting period range."""
    where = "AND is_terminal = FALSE" if active_only else ""
    sql = f"""
        SELECT *
        FROM silver.loan_month_panel
        WHERE reporting_period BETWEEN
            CAST('{start_date}' AS DATE)
            AND CAST('{end_date}' AS DATE)
        {where}
        ORDER BY loan_id, reporting_period
    """
    with get_connection("silver") as conn:
        return conn.execute(sql).pl()


def get_available_reporting_periods() -> list[date]:
    """Return sorted list of all reporting periods in the panel."""
    sql = """
        SELECT DISTINCT reporting_period
        FROM silver.loan_month_panel
        ORDER BY reporting_period
    """
    with get_connection("silver") as conn:
        result = conn.execute(sql).fetchall()
    return [row[0] for row in result]


def get_vintage_range() -> tuple[int, int]:
    """Return (min_vintage_year, max_vintage_year) in the panel."""
    sql = """
        SELECT MIN(origination_year), MAX(origination_year)
        FROM silver.loan_month_panel
    """
    with get_connection("silver") as conn:
        result = conn.execute(sql).fetchone()
    if result and result[0]:
        return int(result[0]), int(result[1])
    return 2012, 2022
