"""Write Bronze layer Parquet files.

Takes a Polars DataFrame and writes it to the Bronze layer
with source metadata columns attached.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stratacredit import DATA_DIR
from stratacredit.ingest.checksums import FileMetadata


def write_bronze(
    df: pl.DataFrame,
    metadata: FileMetadata,
    compression: str = "snappy",
) -> Path:
    """Write a DataFrame to the Bronze Parquet layer.

    Output path:
        data/bronze/{file_type}/{vintage}/{source_file}.parquet

    Attaches source metadata columns to every row:
        _source_file, _source_vintage, _ingested_at, _file_checksum

    Args:
        df: Polars DataFrame from read_origination or read_performance.
        metadata: FileMetadata for this file.
        compression: Parquet compression codec (default 'snappy').

    Returns:
        Path to the written Parquet file.
    """
    df = df.with_columns(
        [
            pl.lit(metadata.source_file).alias("_source_file"),
            pl.lit(metadata.source_vintage).alias("_source_vintage"),
            pl.lit(str(metadata.ingested_at)).alias("_ingested_at"),
            pl.lit(metadata.sha256_checksum).alias("_file_checksum"),
        ]
    )

    out_dir = DATA_DIR / "bronze" / metadata.file_type / metadata.source_vintage
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(metadata.source_file).stem
    out_path = out_dir / f"{stem}.parquet"

    df.write_parquet(out_path, compression=compression)
    return out_path


def bronze_exists(source_file: str, file_type: str, vintage: str) -> bool:
    """Check whether Bronze Parquet already exists for a source file."""
    stem = Path(source_file).stem
    out_path = DATA_DIR / "bronze" / file_type / vintage / f"{stem}.parquet"
    return out_path.exists()
