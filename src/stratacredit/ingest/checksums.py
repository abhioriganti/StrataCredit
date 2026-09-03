"""File checksum and metadata utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FileMetadata:
    """Metadata recorded for every ingested raw file."""

    source_file: str  # filename only
    source_path: str  # full absolute path
    source_vintage: str  # extracted vintage label, e.g. "2015"
    file_type: str  # "origination" or "performance"
    file_size_bytes: int
    sha256_checksum: str
    ingested_at: datetime
    row_count: int = 0


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        path: Path to the file.
        chunk_size: Read chunk size in bytes (default 1 MB).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def collect_metadata(
    path: Path,
    vintage: str,
    file_type: str,
    row_count: int = 0,
) -> FileMetadata:
    """Collect FileMetadata for a raw source file.

    Args:
        path: Absolute path to the raw file.
        vintage: Vintage label (e.g. '2015').
        file_type: 'origination' or 'performance'.
        row_count: Number of rows parsed (filled in after reading).

    Returns:
        FileMetadata instance.
    """
    return FileMetadata(
        source_file=path.name,
        source_path=str(path.resolve()),
        source_vintage=vintage,
        file_type=file_type,
        file_size_bytes=path.stat().st_size,
        sha256_checksum=sha256_file(path),
        ingested_at=datetime.now(tz=timezone.utc),
        row_count=row_count,
    )
