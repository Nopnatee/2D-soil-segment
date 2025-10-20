"""Convenience launcher for the Gradio UI."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from soil_segment.app import run  # noqa: E402  (import after sys.path tweak)


if __name__ == "__main__":
    run()
