# StrataCredit — Data Directory

This directory holds the mortgage loan-level data at four logical processing layers.

**Raw mortgage data files are never committed to Git.**
The `.gitignore` at the repository root excludes all `data/raw/`, `data/bronze/`,
`data/silver/`, and `data/gold/` contents.

---

## Directory Structure

```
data/
├── README.md          — This file
├── raw/               — Original downloaded source files (never modify)
│   └── {vintage}/     — e.g. 2012Q1/, 2012Q2/, ...
├── fixtures/          — Small synthetic datasets for CI/testing
├── bronze/            — Typed Parquet, partitioned by vintage
│   └── origination/
│   └── performance/
├── silver/            — Standardized analytical tables (DuckDB + Parquet)
└── gold/              — Analyst-facing aggregated tables (DuckDB + Parquet)
```

---

## Primary Data Source: Freddie Mac Single-Family Loan-Level Dataset

### Registration

1. Go to: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Register for access (free academic/research registration)
3. Accept the End User License Agreement

### Recommended Download

Download origination and performance files for vintages 2012–2022.

The download portal organizes files as annual or quarterly archives:

```
sample_orig_2012.txt     — Origination: 2012 vintage
sample_svcg_2012.txt     — Performance: 2012 vintage (monthly observations)
sample_orig_2013.txt
sample_svcg_2013.txt
...
sample_orig_2022.txt
sample_svcg_2022.txt
```

### File Placement

Place raw downloaded files in the appropriate vintage subdirectory:

```
data/raw/2012/sample_orig_2012.txt
data/raw/2012/sample_svcg_2012.txt
data/raw/2013/sample_orig_2013.txt
data/raw/2013/sample_svcg_2013.txt
...
```

Or place all files flat in `data/raw/` — the ingestion script will detect vintages
from filenames automatically.

### File Format

- Pipe-delimited (`|`) text files
- No header row
- Column order defined in `configs/data.yaml`
- Encoding: UTF-8
- Sentinels: 9, 99, 999, 9999 used for missing values

---

## Ingestion

After placing raw files, run:

```bash
make ingest
```

This converts raw source files to typed Parquet (Bronze layer) and records:
- source filename
- source vintage
- ingestion timestamp
- SHA-256 file checksum
- file size in bytes

---

## Dataset Scale (after full 2012–2022 ingestion)

Do not hard-code expected row counts.  After ingestion, run:

```bash
python -m stratacredit.ingest.report
```

to see actual counts of:
- unique loans
- loan-month observations
- vintages processed
- voluntary prepayments
- serious delinquencies (90+)
- terminal credit events
- raw storage size
- Bronze Parquet size

---

## Fixtures

The `fixtures/` directory contains synthetic data for testing and CI.
These are small, deterministic datasets with known expected outputs.

Generate fixtures:

```bash
make fixture
```

Fixture files are committed to Git and used in all automated tests.
They are small enough (~500–1,000 synthetic loans) to be fast.

---

## Data License

Freddie Mac Single-Family Loan-Level Dataset is provided under a research
license.  Do not redistribute raw data files.  Do not commit raw data
to any version control system.  See the Freddie Mac EULA for full terms.
