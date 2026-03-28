"""Configuration helpers for filesystem paths.

Reads dataset/checkpoint locations from ``pyproject.toml`` under
``[tool.soil_segment]`` and ensures those directories exist. Paths in the
config may be absolute or relative to the project root.
"""

from dataclasses import dataclass
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class RegressionDatasetSelection:
    path: Path
    uncoated_only: bool = False


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


def append_name_suffix(path: Path, suffix: str) -> Path:
    """Append a suffix to the final path segment name."""
    if not suffix:
        return path
    return path.with_name(f"{path.name}{suffix}")


def _count_image_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        1
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def _has_unet_layout(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "images").is_dir()
        and (path / "masks").is_dir()
    )


def _count_regression_formula_dirs(path: Path, *, uncoated_only: bool = False) -> int:
    if not path.is_dir():
        return 0

    count = 0
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if uncoated_only and "uncoated" not in child.name.lower():
            continue

        images_dir = child / "images" if (child / "images").is_dir() else child
        if _count_image_files(images_dir) > 0:
            count += 1
    return count


def resolve_unet_dataset_path(base_path: Path, *, uncoated: bool = False) -> Path:
    """Resolve the coated/uncoated UNet dataset directory.

    Uncoated mode prefers the conventional ``*_uncoated`` directory, but also
    supports sibling directories such as ``14-7-35-uncoated`` when they expose
    the expected ``images/`` and ``masks/`` structure.
    """
    base_path = Path(base_path).expanduser()
    if not uncoated:
        return base_path

    candidates = [append_name_suffix(base_path, "_uncoated")]
    parent = base_path.parent
    if parent.is_dir():
        for child in sorted(parent.iterdir()):
            if (
                child.is_dir()
                and child not in candidates
                and "uncoated" in child.name.lower()
                and _has_unet_layout(child)
            ):
                candidates.append(child)

    for candidate in candidates:
        if _has_unet_layout(candidate):
            return candidate.resolve()

    return candidates[0]


def resolve_regression_dataset_selection(
    base_path: Path,
    *,
    uncoated: bool = False,
) -> RegressionDatasetSelection:
    """Resolve the regression dataset root for coated/uncoated workflows.

    Uncoated mode supports two layouts:

    1. A dedicated ``regression_dataset_uncoated`` root.
    2. A mixed ``regression_dataset`` root that contains one or more
       ``*-uncoated`` formula subdirectories.
    """
    base_path = Path(base_path).expanduser()
    if not uncoated:
        return RegressionDatasetSelection(path=base_path)

    dedicated = append_name_suffix(base_path, "_uncoated")
    if _count_regression_formula_dirs(dedicated) > 0:
        return RegressionDatasetSelection(path=dedicated.resolve())

    if _count_regression_formula_dirs(base_path, uncoated_only=True) > 0:
        return RegressionDatasetSelection(
            path=base_path.resolve(),
            uncoated_only=True,
        )

    parent = base_path.parent
    if parent.is_dir():
        for child in sorted(parent.iterdir()):
            if (
                child.is_dir()
                and child != dedicated
                and "uncoated" in child.name.lower()
                and _count_regression_formula_dirs(child) > 0
            ):
                return RegressionDatasetSelection(path=child.resolve())

    return RegressionDatasetSelection(path=dedicated)


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
