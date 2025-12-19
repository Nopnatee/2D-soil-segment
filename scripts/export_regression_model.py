"""Export the trained regression model to the configured checkpoints directory."""

import argparse
import shutil
import sys
from pathlib import Path

import joblib

# Make sure the project sources are importable without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soil_segment.config import get_data_paths  # type: ignore

DEFAULT_SOURCE = PROJECT_ROOT / "notebooks" / "checkpoints" / "regression_model.pkl"
DEFAULT_DEST = get_data_paths()["checkpoints"] / "regression_model.pkl"


def export_regression_model(source: Path, destination: Path, *, copy_only: bool = False, force: bool = False) -> Path:
    """Copy or re-serialize the regression checkpoint to a destination path."""
    source = source.expanduser()
    destination = destination.expanduser()

    if not source.exists():
        raise FileNotFoundError(f"Source model not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"Destination already exists: {destination}. Use --force to overwrite.")

    if copy_only:
        shutil.copy2(source, destination)
    else:
        # Load and re-save to verify the checkpoint is readable.
        model = joblib.load(source)
        joblib.dump(model, destination)

    return destination


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or copy the regression model checkpoint to the checkpoints directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Path to an existing regression model (.pkl). Defaults to {DEFAULT_SOURCE}.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination path for the exported model. Defaults to {DEFAULT_DEST}.",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Skip validation and just copy the file instead of loading and re-saving.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        dest_path = export_regression_model(
            args.source,
            args.dest,
            copy_only=args.copy_only,
            force=args.force,
        )
    except Exception as exc:  # pragma: no cover - CLI safety
        print(f"[export-regression] Error: {exc}")
        return 1

    print(f"[export-regression] Exported regression model to: {dest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
