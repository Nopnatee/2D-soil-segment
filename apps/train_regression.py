"""Convenience runner for the regression training workflow."""

from pathlib import Path
import sys
from runpy import run_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


if __name__ == "__main__":
    run_module("soil_segment.train_regression", run_name="__main__")
