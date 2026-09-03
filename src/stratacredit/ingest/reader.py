"""Raw Freddie Mac file reader.

Reads pipe-delimited origination and performance text files
into Polars DataFrames with correct dtypes and sentinel handling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import polars as pl

from stratacredit.ingest.schema import (
    ORIG_COLUMNS,
    ORIG_DTYPE_MAP,
    ORIG_SENTINEL_FLOAT_COLS,
    ORIG_SENTINEL_INT_COLS,
    PERF_COLUMNS,
    PERF_DTYPE_MAP,
    PERF_SENTINEL_FLOAT_COLS,
    PERF_SENTINEL_INT_COLS,
    SENTINEL_FLOAT_VALUES,
    SENTINEL_INT_VALUES,
)

_VINTAGE_PATTERN = re.compile(r"(\d{4})")


def extract_vintage(filename: str) -> str:
    """Extract 4-digit vintage year from a filename.

    e.g. 'sample_orig_2015.txt' -> '2015'
    """
    match = _VINTAGE_PATTERN.search(filename)
    if match:
        return match.group(1)
    return "unknown"


def _replace_sentinels(
    df: pl.DataFrame,
    int_cols: list[str],
    float_cols: list[str],
) -> pl.DataFrame:
    """Replace Freddie Mac sentinel values with null.

    Freddie Mac uses 9, 99, 999, 9999 as missingness codes.
    These must be null before downstream use.
    """
    exprs: list[pl.Expr] = []
    for col in int_cols:
        if col in df.columns:
            exprs.append(
                pl.when(pl.col(col).is_in(SENTINEL_INT_VALUES))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    for col in float_cols:
        if col in df.columns:
            exprs.append(
                pl.when(pl.col(col).is_in(SENTINEL_FLOAT_VALUES))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    if exprs:
        df = df.with_columns(exprs)
    return df


def _read_source_file(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read a headerless Freddie file despite layout-version trailing fields.

    Freddie Mac has added and removed trailing fields across releases.  Read
    the physical width first, map the fields that exist by position, then add
    absent trailing fields as nulls so the Bronze schema remains canonical.
    """
    df = pl.read_csv(
        path,
        separator="|",
        has_header=False,
        infer_schema_length=0,
        null_values=[""],
        truncate_ragged_lines=True,
    )
    # Keep newly released trailing attributes losslessly until their business
    # definitions are promoted into the canonical schema.
    source_columns = columns[: df.width] + [
        f"_source_extension_{position}" for position in range(len(columns) + 1, df.width + 1)
    ]
    df.columns = source_columns
    missing = columns[df.width :]
    if missing:
        df = df.with_columns([pl.lit(None).cast(pl.Utf8).alias(column) for column in missing])
    return df.select(source_columns + missing)


def read_origination(path: Path) -> tuple[pl.DataFrame, int]:
    """Read a Freddie Mac origination file.

    Args:
        path: Path to the pipe-delimited origination text file.

    Returns:
        Tuple of (DataFrame, row_count).
    """
    # Read all columns as Utf8 first to handle mixed formats safely.
    df = _read_source_file(path, ORIG_COLUMNS)

    # Cast to target dtypes
    cast_exprs = []
    for col, dtype in ORIG_DTYPE_MAP.items():
        if col in df.columns:
            if dtype in (pl.Int32, pl.Int64):
                cast_exprs.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))
            elif dtype == pl.Float64:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # Replace remaining sentinels
    df = _replace_sentinels(df, ORIG_SENTINEL_INT_COLS, ORIG_SENTINEL_FLOAT_COLS)

    return df, len(df)


def read_performance(path: Path) -> tuple[pl.DataFrame, int]:
    """Read a Freddie Mac performance file.

    Args:
        path: Path to the pipe-delimited performance text file.

    Returns:
        Tuple of (DataFrame, row_count).
    """
    df = _read_source_file(path, PERF_COLUMNS)

    # Cast numeric columns
    cast_exprs = []
    for col, dtype in PERF_DTYPE_MAP.items():
        if col in df.columns:
            if dtype in (pl.Int32, pl.Int64):
                cast_exprs.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))
            elif dtype == pl.Float64:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # Replace sentinels
    df = _replace_sentinels(df, PERF_SENTINEL_INT_COLS, PERF_SENTINEL_FLOAT_COLS)

    return df, len(df)


def detect_file_type(filename: str) -> Literal["origination", "performance", "unknown"]:
    """Detect whether a file is origination or performance based on name."""
    name = filename.lower()
    if "orig" in name:
        return "origination"
    if "svcg" in name or "perf" in name:
        return "performance"
    return "unknown"
