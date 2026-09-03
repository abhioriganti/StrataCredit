"""Transform module runner entry point."""

import sys

from stratacredit.transform.builder import run

if __name__ == "__main__":
    run(force="--force" in sys.argv)
