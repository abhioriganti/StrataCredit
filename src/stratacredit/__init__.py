"""StrataCredit: Loan-Level Structured Credit Analytics & Pool Selection Platform."""

__version__ = "0.1.0"
__author__ = "StrataCredit"

from pathlib import Path

# Repository root
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
SQL_DIR = ROOT_DIR / "sql"
