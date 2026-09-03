"""Pool eligibility filters.

Applies hard eligibility criteria to produce a candidate loan universe.
A loan failing any eligibility filter is excluded before constraint
optimization begins.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from stratacredit.config import pool_config


@dataclass
class EligibilityConfig:
    """Eligibility filter parameters."""

    require_current: bool = True
    exclude_terminated: bool = True
    exclude_active_modifications: bool = False
    exclude_ever_90dpd: bool = False
    min_remaining_term: int = 12
    min_upb: float = 10_000.0
    max_upb: float = 2_000_000.0
    vintage_min: int | None = None
    vintage_max: int | None = None
    allowed_property_types: list[str] | None = None
    allowed_occupancy_types: list[str] | None = None

    @classmethod
    def from_config(cls) -> EligibilityConfig:
        """Load from pool_constraints.yaml."""
        e = pool_config["eligibility"]
        return cls(
            require_current=e.get("require_current", True),
            exclude_terminated=e.get("exclude_terminated", True),
            exclude_active_modifications=e.get("exclude_active_modifications", False),
            exclude_ever_90dpd=e.get("exclude_ever_90dpd", False),
            min_remaining_term=e.get("min_remaining_term", 12),
            min_upb=e.get("min_upb", 10_000.0),
            max_upb=e.get("max_upb", 2_000_000.0),
            vintage_min=e.get("vintage_min"),
            vintage_max=e.get("vintage_max"),
            allowed_property_types=e.get("allowed_property_types"),
            allowed_occupancy_types=e.get("allowed_occupancy_types"),
        )


def apply_eligibility_filters(
    df: pl.DataFrame,
    config: EligibilityConfig | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Apply eligibility filters and return candidate pool with exclusion stats.

    Args:
        df: Snapshot of loan_month_panel (one reporting period).
        config: Eligibility configuration. Loads from YAML if None.

    Returns:
        (candidate_df, exclusion_counts dict)
    """
    cfg = config or EligibilityConfig.from_config()
    exclusions: dict[str, int] = {}
    original_n = len(df)

    if cfg.exclude_terminated:
        mask = pl.col("is_terminal") == False  # noqa: E712
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["terminated"] = excl
        df = df.filter(mask)

    if cfg.require_current:
        mask = pl.col("delinquency_state") == "CURRENT"
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["not_current"] = excl
        df = df.filter(mask)

    if "remaining_term" in df.columns:
        mask = (pl.col("remaining_term").is_null()) | (
            pl.col("remaining_term") >= cfg.min_remaining_term
        )
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["short_remaining_term"] = excl
        df = df.filter(mask)

    if "current_upb" in df.columns:
        mask = (pl.col("current_upb") >= cfg.min_upb) & (pl.col("current_upb") <= cfg.max_upb)
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["upb_out_of_range"] = excl
        df = df.filter(mask)

    if cfg.vintage_min and "origination_year" in df.columns:
        mask = pl.col("origination_year") >= cfg.vintage_min
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["vintage_too_old"] = excl
        df = df.filter(mask)

    if cfg.vintage_max and "origination_year" in df.columns:
        mask = pl.col("origination_year") <= cfg.vintage_max
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["vintage_too_new"] = excl
        df = df.filter(mask)

    if cfg.allowed_property_types and "property_type" in df.columns:
        mask = pl.col("property_type").is_in(cfg.allowed_property_types)
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["disallowed_property_type"] = excl
        df = df.filter(mask)

    if cfg.allowed_occupancy_types and "occupancy" in df.columns:
        mask = pl.col("occupancy").is_in(cfg.allowed_occupancy_types)
        excl = len(df) - len(df.filter(mask))
        if excl > 0:
            exclusions["disallowed_occupancy"] = excl
        df = df.filter(mask)

    exclusions["_total_excluded"] = original_n - len(df)
    exclusions["_candidate_count"] = len(df)
    return df, exclusions
