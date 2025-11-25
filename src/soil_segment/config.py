"""Configuration helpers for filesystem paths.

Reads dataset/checkpoint locations from ``pyproject.toml`` under
``[tool.soil_segment]`` and ensures those directories exist. Paths in the
config may be absolute or relative to the project root.
"""

from pathlib import Path
from typing import Dict

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Pythons
    import tomli as tomllib  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

DEFAULT_UNET_DATASET = PROJECT_ROOT / "datasets" / "UNET_dataset"
DEFAULT_REGRESSION_DATASET = PROJECT_ROOT / "datasets" / "regression_dataset"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def _safe_load_config() -> Dict[str, str]:
    if not PYPROJECT_PATH.exists():
        return {}

    try:
        data = tomllib.loads(PYPROJECT_PATH.read_text())
        return data.get("tool", {}).get("soil_segment", {}) or {}
    except Exception:
        # Do not block execution if config parsing fails; fall back to defaults.
        return {}


def _resolve_path(value, default: Path) -> Path:
    path = Path(value) if value is not None else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_paths() -> Dict[str, Path]:
    """Return project data paths, ensuring the directories exist."""

    cfg = _safe_load_config()

    unet_dataset = _resolve_path(cfg.get("unet_dataset"), DEFAULT_UNET_DATASET)
    regression_dataset = _resolve_path(
        cfg.get("regression_dataset"), DEFAULT_REGRESSION_DATASET
    )
    checkpoints_dir = _resolve_path(
        cfg.get("checkpoints_dir"), DEFAULT_CHECKPOINTS_DIR
    )

    return {
        "unet_dataset": unet_dataset,
        "regression_dataset": regression_dataset,
        "checkpoints": checkpoints_dir,
    }
