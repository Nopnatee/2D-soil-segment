from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - UI dependency
    raise SystemExit(
        "OpenCV (cv2) is required for the annotator. Install with: pip install opencv-python"
    ) from exc

import torch
import torch.nn.functional as F

from soil_segment.custom_unet import SimpleUNet  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("datasets/raw_images")
DEFAULT_OUTPUT_DATASET = Path("datasets/annotated_dataset")


def _default_checkpoint_path() -> Path:
    """Return repo-root-relative default checkpoint path.

    Prefers `checkpoints/best_model.pth` next to the project root regardless
    of current working directory.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "checkpoints" / "best_model.pth"


def _ensure_dirs(dataset_root: Path) -> Tuple[Path, Path, Path]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    overlays_dir = dataset_root / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir, overlays_dir


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])


def _default_palette(n: int) -> np.ndarray:
    """Return an Nx3 uint8 color palette for class visualization."""
    base = np.array(
        [
            [0, 0, 0],        # 0: background
            [220, 20, 60],    # 1
            [0, 128, 0],      # 2
            [30, 144, 255],   # 3
            [255, 140, 0],    # 4
            [138, 43, 226],   # 5
            [255, 215, 0],    # 6
        ],
        dtype=np.uint8,
    )
    if n <= base.shape[0]:
        return base[:n]

    # Generate additional colors deterministically
    rng = np.random.default_rng(12345)
    extra = rng.integers(0, 256, size=(n - base.shape[0], 3), dtype=np.uint8)
    return np.vstack([base, extra])


def _render_overlay(image_bgr: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized mask onto the image for visualization."""
    h, w = mask.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(mask):
        if cid >= len(palette):
            continue
        color_mask[mask == cid] = palette[cid]
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0)
    return overlay

# ---------------------------------------------------------------------------
# Automated annotation (model inference)
# ---------------------------------------------------------------------------

def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, map_location: str | torch.device) -> None:
    payload = torch.load(str(ckpt_path), map_location=map_location)

    if isinstance(payload, dict):
        # Common patterns: 'model_state_dict', 'state_dict', or full model in 'model'
        state = None
        for key in ("model_state_dict", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
        if state is None and all(isinstance(k, str) for k in payload.keys()):
            # Might be a raw state_dict
            state = payload  # type: ignore[assignment]

        if state is None and hasattr(payload, "state_dict"):
            state = payload.state_dict()

        if state is None:
            raise RuntimeError("Unrecognized checkpoint format: missing state_dict.")

        # Strip 'module.' if saved from DataParallel
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)
    else:
        # Rare case: entire model object saved
        try:
            model.load_state_dict(payload.state_dict())  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unsupported checkpoint object.") from exc


def _prepare_image_tensor(img_bgr: np.ndarray, target_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    # Convert BGR -> RGB, resize to square
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # To float tensor [0,1]
    x = resized.astype(np.float32) / 255.0
    # Normalize as in trainer
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    # HWC -> CHW
    x = np.transpose(x, (0, 1, 2))
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x, (h, w)


def _predict_mask(model: torch.nn.Module, x: torch.Tensor, num_classes: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if logits.shape[1] != num_classes:
            # If checkpoint trained with different num_classes, keep argmax anyway
            pass
        pred = torch.argmax(logits, dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)
    return pred


def auto_annotate(
    *,
    input_dir: Path,
    output_dataset: Path,
    checkpoint: Path,
    num_classes: int = 7,
    class_names: Optional[List[str]] = None,
    img_size: int = 1024,
    device_str: Optional[str] = None,
    skip_existing: bool = True,
    save_overlays: bool = True,
) -> None:
    images_dir, masks_dir, overlays_dir = _ensure_dirs(output_dataset)
    input_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}.")

    # Save class metadata
    meta = {
        "num_classes": num_classes,
        "class_names": class_names or [f"Class {i}" for i in range(num_classes)],
    }
    (output_dataset / "classes.json").write_text(json.dumps(meta, indent=2))

    # Device
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build and load model
    model = SimpleUNet(n_classes=num_classes)
    _load_checkpoint(model, checkpoint, map_location=device)
    model.to(device)

    palette = _default_palette(num_classes)

    for idx, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[warn] Failed to read image: {img_path}")
            continue

        # Copy the original image into dataset/images
        dst_img_path = images_dir / img_path.name
        if not dst_img_path.exists():
            cv2.imwrite(str(dst_img_path), img_bgr)

        dst_mask_path = masks_dir / f"{img_path.stem}.png"
        if skip_existing and dst_mask_path.exists():
            print(f"[{idx}/{len(images)}] Skip existing mask: {dst_mask_path.name}")
            continue

        x, (h, w) = _prepare_image_tensor(img_bgr, img_size)
        pred_resized = _predict_mask(model, x, num_classes, device)
        # Resize prediction back to original resolution (nearest)
        pred_mask = cv2.resize(pred_resized, (w, h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(dst_mask_path), pred_mask)
        if save_overlays:
            overlay = _render_overlay(img_bgr, pred_mask, palette)
            cv2.imwrite(str((overlays_dir / f"{img_path.stem}.png")), overlay)

        print(f"[{idx}/{len(images)}] Wrote mask: {dst_mask_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segmentation annotator: automated model-based pre-labeling")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory of raw images to annotate.")
    p.add_argument(
        "--output-dataset",
        type=Path,
        default=DEFAULT_OUTPUT_DATASET,
        help="Root dataset directory where images/masks/overlays will be saved.",
    )
    p.add_argument("--num-classes", type=int, default=7, help="Number of classes (IDs 0..N-1).")
    p.add_argument("--class-names", nargs="*", help="Optional class names; must match --num-classes if provided.")

    p.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Path to trained model checkpoint (.pt/.pth). "
            "If omitted, uses checkpoints/best_model.pth when present."
        ),
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="Inference size the model expects (square, match training).",
    )
    p.add_argument("--device", type=str, help="cpu or cuda (auto if omitted).")
    p.add_argument("--no-skip-existing", action="store_true", help="Recompute masks even if they exist.")
    p.add_argument("--no-overlays", action="store_true", help="Do not save overlay previews.")
    return p.parse_args()


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - entry point
    args = parse_args()
    # Ensure raw input dir exists to guide the user
    args.input_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint:
        default_ckpt = _default_checkpoint_path()
        if default_ckpt.exists():
            print(f"[info] Using default checkpoint: {default_ckpt}")
            args.checkpoint = default_ckpt
        else:
            raise SystemExit(
                "--checkpoint is required (manual annotation has been removed). "
                "Place a model at checkpoints/best_model.pth or pass --checkpoint."
            )

    # Automated path
    auto_annotate(
        input_dir=args.input_dir,
        output_dataset=args.output_dataset,
        checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        class_names=args.class_names,
        img_size=args.img_size,
        device_str=args.device,
        skip_existing=not args.no_skip_existing,
        save_overlays=not args.no_overlays,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
