import argparse
from pathlib import Path
from typing import List, Tuple

ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}


def find_mask_candidate(mask_dir: Path, stem: str, suffixes: List[str]) -> Tuple[Path, str]:
    """Return the first existing mask path for a given image stem.

    Tries patterns in order:
      - <stem>.<ext>
      - <stem><suffix>.<ext> for each suffix in suffixes

    Returns (Path, ext) or (None, "") if not found.
    """
    # Exact stem match first
    for ext in ALLOWED_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p, ext

    # Then try suffix patterns
    for suffix in suffixes:
        for ext in ALLOWED_EXTS:
            p = mask_dir / f"{stem}{suffix}{ext}"
            if p.exists():
                return p, ext

    return Path(), ""


def rename_masks(images_dir: Path, masks_dir: Path, suffixes: List[str], overwrite: bool, dry_run: bool) -> None:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]

    renamed = 0
    skipped_exists = 0
    missing = 0
    already_ok = 0

    for img in images:
        stem = img.stem
        src_mask, ext = find_mask_candidate(masks_dir, stem, suffixes)

        if not ext:
            print(f"[MISS] No mask found for image: {img.name}")
            missing += 1
            continue

        target = masks_dir / f"{stem}{ext}"

        if src_mask == target:
            # Already correctly named
            already_ok += 1
            continue

        if target.exists() and not overwrite:
            print(f"[SKIP] Target exists, not overwriting: {target.name}")
            skipped_exists += 1
            continue

        action = "Would rename" if dry_run else "Renaming"
        print(f"[{action}] {src_mask.name} -> {target.name}")
        if not dry_run:
            # Ensure parent exists
            target.parent.mkdir(parents=True, exist_ok=True)
            src_mask.replace(target)
        renamed += 1

    print("\nSummary:")
    print(f"  Images scanned     : {len(images)}")
    print(f"  Renamed masks      : {renamed}")
    print(f"  Already correct    : {already_ok}")
    print(f"  Skipped (exists)   : {skipped_exists}")
    print(f"  Missing masks      : {missing}")


def main():
    parser = argparse.ArgumentParser(description="Rename mask files to match image stems.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Path to images directory")
    parser.add_argument("--masks-dir", type=Path, required=True, help="Path to masks directory")
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=["_mask", "-mask"],
        help="Mask suffixes to try (default: _mask -mask)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without renaming")
    args = parser.parse_args()

    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise SystemExit(f"Images dir not found or not a directory: {args.images_dir}")
    if not args.masks_dir.exists() or not args.masks_dir.is_dir():
        raise SystemExit(f"Masks dir not found or not a directory: {args.masks_dir}")

    rename_masks(args.images_dir, args.masks_dir, args.suffixes, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()

