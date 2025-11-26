from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT if SCRIPT_ROOT.name != "scripts" else SCRIPT_ROOT.parent
DEFAULT_DEST = REPO_ROOT / "datasets" / "UNET_dataset"

DEFAULT_WORKSPACE = "npk-segmentation"
DEFAULT_PROJECT = "npk-segmentation-d40wm"
DEFAULT_VERSION = 7
DEFAULT_FORMAT = "png-mask-semantic"
DEFAULT_PREFIX = "img."

def _load_env_file() -> Optional[Path]:
    """Lightweight .env loader to prime os.environ for defaults."""
    for candidate_dir in (SCRIPT_ROOT,) + tuple(SCRIPT_ROOT.parents):
        env_path = candidate_dir / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return env_path
    return None


def _ensure_roboflow():
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError as exc:  # pragma: no cover - convenience guard
        raise SystemExit(
            "roboflow is not installed. Install it with `pip install roboflow`."
        ) from exc
    return Roboflow


def _dataset_location(dataset) -> Path:
    location = getattr(dataset, "location", None) or getattr(dataset, "path", None)
    if not location:
        raise RuntimeError("Could not determine download location from Roboflow response.")
    return Path(location)


def download_dataset(api_key: str, workspace: str, project: str, version: int, export_format: str) -> Path:
    Roboflow = _ensure_roboflow()
    rf = Roboflow(api_key=api_key)
    project_ref = rf.workspace(workspace).project(project)
    dataset = project_ref.version(version).download(export_format)
    return _dataset_location(dataset)


def _is_mask_path(path: Path) -> bool:
    names = {part.lower() for part in path.parts}
    return any(token in {"labels", "label", "masks", "mask"} for token in names)


def _collect_pairs(source_root: Path) -> Tuple[List[Tuple[Path, Path]], List[str], List[str]]:
    image_map: Dict[str, Path] = {}
    mask_map: Dict[str, Path] = {}
    valid_exts = {".jpg", ".jpeg", ".png"}

    for file_path in source_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in valid_exts:
            continue

        stem = file_path.stem
        parent_name = file_path.parent.name.lower()

        if parent_name == "images" or ("image" in parent_name and not _is_mask_path(file_path.parent)):
            image_map[stem] = file_path
            continue

        if parent_name in {"labels", "masks", "mask"} or _is_mask_path(file_path.parent):
            mask_map[stem] = file_path
            continue

        # Fallback: if we already saw a mask with this stem, treat as image; otherwise assume image first
        if stem in mask_map:
            image_map[stem] = file_path
        else:
            image_map.setdefault(stem, file_path)

    common = sorted(set(image_map) & set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    missing_masks = sorted(set(image_map) - set(mask_map))

    pairs = [(image_map[name], mask_map[name]) for name in common]
    return pairs, missing_images, missing_masks


def _prepare_output_dirs(dest: Path, clean: bool) -> Tuple[Path, Path]:
    images_dir = dest / "images"
    masks_dir = dest / "masks"
    for target in (images_dir, masks_dir):
        target.mkdir(parents=True, exist_ok=True)
        if clean:
            for child in target.iterdir():
                if child.is_file():
                    child.unlink()
    return images_dir, masks_dir


def _copy_pairs(
    pairs: Iterable[Tuple[Path, Path]],
    images_dir: Path,
    masks_dir: Path,
    prefix: str,
    start_index: int,
) -> int:
    pairs_list = list(pairs)
    total = len(pairs_list)
    pad_width = max(3, len(str(start_index + total - 1)))

    for idx, (image_path, mask_path) in enumerate(pairs_list, start=start_index):
        new_stem = f"{prefix}{idx:0{pad_width}d}"
        dest_image = images_dir / f"{new_stem}{image_path.suffix.lower()}"
        dest_mask = masks_dir / f"{new_stem}{mask_path.suffix.lower()}"
        shutil.copy2(image_path, dest_image)
        shutil.copy2(mask_path, dest_mask)

    return total


def _build_defaults(env_dir: Path) -> Dict[str, object]:
    """Gather defaults from environment (preloaded via .env) with fallbacks."""
    dest_env = os.getenv("ROBOFLOW_DEST") or os.getenv("ROBOFLOW_PATH")
    dest_path = Path(dest_env) if dest_env else DEFAULT_DEST
    if not dest_path.is_absolute():
        dest_path = env_dir / dest_path

    version_env = os.getenv("ROBOFLOW_VERSION")
    try:
        version_val = int(version_env) if version_env is not None else DEFAULT_VERSION
    except ValueError:
        version_val = DEFAULT_VERSION

    return {
        "api_key": os.getenv("ROBOFLOW_API_KEY"),
        "workspace": os.getenv("ROBOFLOW_WORKSPACE", DEFAULT_WORKSPACE),
        "project": os.getenv("ROBOFLOW_PROJECT", DEFAULT_PROJECT),
        "version": version_val,
        "format": os.getenv("ROBOFLOW_FORMAT", DEFAULT_FORMAT),
        "dest": dest_path,
    }


def parse_args(argv: List[str], defaults: Dict[str, object]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Roboflow NPK segmentation dataset and normalize file names."
    )
    parser.add_argument("--api-key", default=defaults["api_key"], help="Roboflow API key or set ROBOFLOW_API_KEY.")
    parser.add_argument("--workspace", default=defaults["workspace"], help="Roboflow workspace slug.")
    parser.add_argument("--project", default=defaults["project"], help="Roboflow project slug.")
    parser.add_argument("--version", type=int, default=defaults["version"], help="Dataset version to download.")
    parser.add_argument("--format", default=defaults["format"], help="Roboflow export format.")
    parser.add_argument("--dest", type=Path, default=defaults["dest"], help="Where to place normalized images and masks.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefix for renamed files (default: img.).")
    parser.add_argument("--start-index", type=int, default=1, help="Starting index for renamed files.")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Use an existing Roboflow export directory instead of downloading.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing files in dest/images and dest/masks instead of clearing them first.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    env_path = _load_env_file()
    env_dir = env_path.parent if env_path else REPO_ROOT
    defaults = _build_defaults(env_dir)

    args = parse_args(argv or sys.argv[1:], defaults)

    if not args.source_root and not args.api_key:
        raise SystemExit(
            "Provide --api-key (or set ROBOFLOW_API_KEY). A .env file was "
            f"{'found' if env_path else 'not found'} in the expected locations."
        )

    # Resolve destination relative to the .env location (or script dir) if not absolute
    if not args.dest.is_absolute():
        args.dest = (env_dir / args.dest).resolve()

    source_root = args.source_root
    if source_root is None:
        print("Downloading dataset from Roboflow...")
        source_root = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            export_format=args.format,
        )
        print(f"Download complete: {source_root}")

    if not source_root.exists():
        raise SystemExit(f"Source root '{source_root}' does not exist.")

    pairs, missing_images, missing_masks = _collect_pairs(source_root)
    if not pairs:
        raise SystemExit("No image/mask pairs found. Check the download contents.")

    images_dir, masks_dir = _prepare_output_dirs(args.dest, clean=not args.no_clean)
    total_copied = _copy_pairs(
        pairs=pairs,
        images_dir=images_dir,
        masks_dir=masks_dir,
        prefix=args.prefix,
        start_index=args.start_index,
    )

    print(f"Copied {total_copied} pairs to '{args.dest}'.")
    if missing_images:
        print(f"Warning: {len(missing_images)} masks had no matching image: {missing_images[:5]}")
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images had no matching mask: {missing_masks[:5]}")
    print(f"Examples saved to {images_dir} and {masks_dir}")


if __name__ == "__main__":
    main()
