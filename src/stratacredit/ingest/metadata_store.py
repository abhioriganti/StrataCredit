"""Persistent ingestion metadata store.

Tracks every ingested file with checksum, row count, and timestamp
in a JSON-lines file.  Enables idempotent re-ingestion and audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from stratacredit import DATA_DIR
from stratacredit.ingest.checksums import FileMetadata

_STORE_PATH = DATA_DIR / "bronze" / "_ingestion_log.jsonl"


def append_metadata(metadata: FileMetadata) -> None:
    """Append a FileMetadata record to the ingestion log."""
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(metadata)
    record["ingested_at"] = metadata.ingested_at.isoformat()
    with open(_STORE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_metadata() -> list[dict]:
    """Load all ingestion metadata records."""
    if not _STORE_PATH.exists():
        return []
    records = []
    with open(_STORE_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_ingested(sha256: str) -> bool:
    """Check whether a file with a given checksum was already ingested."""
    return any(rec.get("sha256_checksum") == sha256 for rec in load_metadata())


def ingestion_summary() -> dict:
    """Return a summary of all ingested files."""
    records = load_metadata()
    return {
        "total_files": len(records),
        "origination_files": sum(1 for r in records if r.get("file_type") == "origination"),
        "performance_files": sum(1 for r in records if r.get("file_type") == "performance"),
        "total_rows": sum(r.get("row_count", 0) for r in records),
        "vintages": sorted({r.get("source_vintage", "unknown") for r in records}),
    }
