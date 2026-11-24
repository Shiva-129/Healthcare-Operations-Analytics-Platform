"""
main.py
-------
Orchestrates the full ETL pipeline:
    Extract -> Transform -> Load

Run with:
    python src/main.py
"""

import logging
import sys
import time
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.extract import main as extract_main
from src.transform import run_transform
from src.load_sql import main as load_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("etl_orchestrator")


def run_pipeline():
    print("=" * 60)
    print("HOSPITAL ANALYTICS ETL PIPELINE")
    print("=" * 60)

    start = time.time()

    print("\n[1/3] EXTRACT")
    print("-" * 60)
    extract_main()

    print("\n[2/3] TRANSFORM")
    print("-" * 60)
    run_transform()

    print("\n[3/3] LOAD")
    print("-" * 60)
    load_main()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE in {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
