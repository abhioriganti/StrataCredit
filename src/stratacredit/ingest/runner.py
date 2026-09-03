"""Ingestion pipeline runner.

Entry point for `make ingest` / `python -m stratacredit.ingest.runner`.

Scans data/raw/ for Freddie Mac origination and performance files,
reads each file, writes Bronze Parquet, and records metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from stratacredit import DATA_DIR
from stratacredit.ingest.bronze_writer import bronze_exists, write_bronze
from stratacredit.ingest.checksums import collect_metadata, sha256_file
from stratacredit.ingest.metadata_store import (
    already_ingested,
    append_metadata,
    ingestion_summary,
)
from stratacredit.ingest.reader import (
    detect_file_type,
    extract_vintage,
    read_origination,
    read_performance,
)

console = Console()


def discover_raw_files(raw_dir: Path) -> list[Path]:
    """Find all .txt source files in the raw directory tree."""
    return sorted(raw_dir.rglob("*.txt"))


def ingest_file(path: Path, force: bool = False) -> bool:
    """Ingest a single raw source file to Bronze.

    Args:
        path: Path to the raw .txt file.
        force: Re-ingest even if already present.

    Returns:
        True if file was ingested, False if skipped.
    """
    file_type = detect_file_type(path.name)
    if file_type == "unknown":
        console.print(f"  [yellow]SKIP[/yellow] {path.name} — unrecognised file type")
        return False

    vintage = extract_vintage(path.name)
    checksum = sha256_file(path)

    if not force:
        if already_ingested(checksum):
            console.print(f"  [dim]SKIP[/dim] {path.name} — already ingested")
            return False
        if bronze_exists(path.name, file_type, vintage):
            console.print(f"  [dim]SKIP[/dim] {path.name} — Bronze already exists")
            return False

    console.print(f"  [cyan]READ[/cyan] {path.name}  (vintage={vintage}, type={file_type})")

    if file_type == "origination":
        df, row_count = read_origination(path)
    else:
        df, row_count = read_performance(path)

    metadata = collect_metadata(path, vintage, file_type, row_count=row_count)
    metadata.sha256_checksum = checksum

    out_path = write_bronze(df, metadata)
    append_metadata(metadata)

    console.print(f"  [green]DONE[/green] {row_count:,} rows → {out_path.relative_to(DATA_DIR)}")
    return True


def run(raw_dir: Path | None = None, force: bool = False) -> None:
    """Run the full ingestion pipeline.

    Args:
        raw_dir: Override for the raw data directory.
        force: Force re-ingestion of all files.
    """
    raw_dir = raw_dir or DATA_DIR / "raw"

    console.rule("[bold blue]StrataCredit Ingestion Pipeline")
    console.print(f"Scanning: {raw_dir}")

    files = discover_raw_files(raw_dir)
    if not files:
        console.print("[yellow]No .txt files found in data/raw/.  Nothing to ingest.[/yellow]")
        console.print("See data/README.md for download instructions.")
        return

    console.print(f"Found {len(files)} file(s) to process.\n")

    ingested = 0
    skipped = 0
    for f in files:
        ok = ingest_file(f, force=force)
        if ok:
            ingested += 1
        else:
            skipped += 1

    console.print()
    summary = ingestion_summary()
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Files ingested this run", str(ingested))
    table.add_row("Files skipped", str(skipped))
    table.add_row("Total files on record", str(summary["total_files"]))
    table.add_row("Total rows on record", f"{summary['total_rows']:,}")
    table.add_row("Vintages on record", ", ".join(summary["vintages"]) or "none")
    console.print(table)


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run(force=force_flag)
