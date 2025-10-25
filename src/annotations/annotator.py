"""Lightweight local image annotator for segmentation masks.

This tool paints single-channel masks with class IDs that match the
training pipeline (see `src/soil_segment/trainer.py`). Masks are saved
under `datasets/` with the same filename as the source image, and can be
uploaded later using `src/annotations/roboflow_uploader.py`.

Key goals:
- Keep class IDs (pixel values) consistent with the trainer (0..N-1).
- Separate concerns: this module ONLY annotates and saves; uploading is
  handled by `roboflow_uploader.py`.
- Simple OpenCV-based UI for brush painting and quick iteration.

Usage examples:
  python -m annotations.annotator \
      --input-dir datasets/raw_images \
      --output-dataset datasets/UNET_dataset \
      --num-classes 7 \
      --class-names Background Sand Clay Silt Rock Water Organic

Controls (interactive mode):
- Mouse left button: paint with current class
- Mouse right button: erase to class 0 (background)
- 0..9 keys: set current class by digit
- [ / ]: decrease / increase brush size
- c / v: previous / next class
- z: undo last stroke
- s: save mask + overlay
- n / p: next / previous image
- q or ESC: quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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

try:
    from soil_segment.custom_unet import SimpleUNet  # type: ignore
except ModuleNotFoundError:  # allow running from repo root without install
    here = Path(__file__).resolve().parents[2]
    src_path = here / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from soil_segment.custom_unet import SimpleUNet  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("datasets/raw_images")
DEFAULT_OUTPUT_DATASET = Path("datasets/UNET_dataset")


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
            [0, 206, 209],    # 7
            [205, 92, 92],    # 8
            [154, 205, 50],   # 9
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


@dataclass
class SessionState:
    images: List[Path]
    idx: int
    brush: int
    current_class: int
    num_classes: int
    class_names: List[str]
    palette: np.ndarray
    undo_buffer: Optional[np.ndarray] = None


class MaskPainter:
    def __init__(
        self,
        input_dir: Path,
        output_dataset: Path,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        brush_size: int = 10,
    ) -> None:
        self.input_dir = input_dir
        self.output_dataset = output_dataset
        self.images_dir, self.masks_dir, self.overlays_dir = _ensure_dirs(output_dataset)

        images = _list_images(input_dir)
        if not images:
            raise FileNotFoundError(f"No images found in {input_dir}.")

        if class_names and len(class_names) != num_classes:
            raise ValueError("Length of --class-names must match --num-classes.")

        self.state = SessionState(
            images=images,
            idx=0,
            brush=max(1, brush_size),
            current_class=1 if num_classes > 1 else 0,
            num_classes=num_classes,
            class_names=class_names or [f"Class {i}" for i in range(num_classes)],
            palette=_default_palette(num_classes),
        )

        # Save class metadata for downstream use
        meta = {
            "num_classes": num_classes,
            "class_names": self.state.class_names,
        }
        (self.output_dataset / "classes.json").write_text(json.dumps(meta, indent=2))

        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotator", self._on_mouse)

        self._load_current()

    # ------------------------------- IO ---------------------------------- #
    def _img_path(self, idx: int) -> Path:
        return self.state.images[idx]

    def _mask_path(self, idx: int) -> Path:
        stem = self._img_path(idx).stem
        return self.masks_dir / f"{stem}.png"

    def _overlay_path(self, idx: int) -> Path:
        stem = self._img_path(idx).stem
        return self.overlays_dir / f"{stem}.png"

    def _load_current(self) -> None:
        img_path = self._img_path(self.state.idx)
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        self.image_bgr = image_bgr

        # Copy source image into dataset/images if not already present
        dst = self.images_dir / img_path.name
        if not dst.exists():
            cv2.imwrite(str(dst), image_bgr)

        h, w = image_bgr.shape[:2]
        mask_path = self._mask_path(self.state.idx)
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                # Fallback if saved with color by mistake
                mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        self.mask = mask.astype(np.uint8)
        self.state.undo_buffer = None

    def _save_current(self) -> None:
        # Save mask as single-channel PNG with class IDs
        cv2.imwrite(str(self._mask_path(self.state.idx)), self.mask)
        # Save overlay preview for quick browsing
        overlay = _render_overlay(self.image_bgr, self.mask, self.state.palette)
        cv2.imwrite(str(self._overlay_path(self.state.idx)), overlay)

    # ----------------------------- Painting ------------------------------- #
    def _on_mouse(self, event: int, x: int, y: int, flags: int, userdata=None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.state.undo_buffer = self.mask.copy()

        if flags & cv2.EVENT_FLAG_LBUTTON:
            self._paint_at(x, y, self.state.current_class)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            self._paint_at(x, y, 0)

    def _paint_at(self, x: int, y: int, cls: int) -> None:
        radius = max(1, int(self.state.brush))
        cv2.circle(self.mask, (x, y), radius, int(cls), thickness=-1)

    # ------------------------------ UI Loop ------------------------------- #
    def run(self) -> None:  # pragma: no cover - interactive loop
        while True:
            vis = _render_overlay(self.image_bgr, self.mask, self.state.palette)
            self._draw_hud(vis)
            cv2.imshow("Annotator", vis)
            key = cv2.waitKey(15) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or q
                break
            elif key == ord('s'):
                self._save_current()
            elif key == ord('n'):
                self._next_image()
            elif key == ord('p'):
                self._prev_image()
            elif key == ord('z') and self.state.undo_buffer is not None:
                self.mask = self.state.undo_buffer
                self.state.undo_buffer = None
            elif key == ord('['):
                self.state.brush = max(1, self.state.brush - 1)
            elif key == ord(']'):
                self.state.brush = min(256, self.state.brush + 1)
            elif key == ord('c'):
                self.state.current_class = (self.state.current_class - 1) % self.state.num_classes
            elif key == ord('v'):
                self.state.current_class = (self.state.current_class + 1) % self.state.num_classes
            elif key in range(ord('0'), ord('9') + 1):
                digit = key - ord('0')
                if digit < self.state.num_classes:
                    self.state.current_class = digit

        cv2.destroyAllWindows()

    def _next_image(self) -> None:
        if self.state.idx < len(self.state.images) - 1:
            self._save_current()
            self.state.idx += 1
            self._load_current()

    def _prev_image(self) -> None:
        if self.state.idx > 0:
            self._save_current()
            self.state.idx -= 1
            self._load_current()

    def _draw_hud(self, vis: np.ndarray) -> None:
        h, w = vis.shape[:2]
        y = 24
        pad = 8
        # Info text
        info = (
            f"[{self.state.idx + 1}/{len(self.state.images)}] "
            f"Class: {self.state.current_class} ({self.state.class_names[self.state.current_class]})  "
            f"Brush: {self.state.brush}px  "
            f"s=save n/p=next/prev [ ]=brush 0-9=set c/v=class z=undo q=quit"
        )
        cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), thickness=-1)
        cv2.putText(vis, info, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Legend swatches
        sw = 20
        for cid in range(min(self.state.num_classes, 10)):
            color = tuple(int(c) for c in self.state.palette[cid].tolist())
            x0 = pad + cid * (sw + 6)
            cv2.rectangle(vis, (x0, 48), (x0 + sw, 48 + sw), color, thickness=-1)
            cv2.putText(vis, str(cid), (x0 + 4, 48 + sw + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
    img_size: int = 512,
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
    p = argparse.ArgumentParser(description="Segmentation annotator: automated (default) and interactive modes")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory of raw images to annotate.")
    p.add_argument(
        "--output-dataset",
        type=Path,
        default=DEFAULT_OUTPUT_DATASET,
        help="Root dataset directory where images/masks/overlays will be saved.",
    )
    p.add_argument("--num-classes", type=int, default=7, help="Number of classes (IDs 0..N-1).")
    p.add_argument("--class-names", nargs="*", help="Optional class names; must match --num-classes if provided.")

    # Automated options
    p.add_argument("--checkpoint", type=Path, help="Path to trained model checkpoint (.pt/.pth).")
    p.add_argument("--img-size", type=int, default=512, help="Inference size the model expects (square).")
    p.add_argument("--device", type=str, help="cpu or cuda (auto if omitted).")
    p.add_argument("--no-skip-existing", action="store_true", help="Recompute masks even if they exist.")
    p.add_argument("--no-overlays", action="store_true", help="Do not save overlay previews.")

    # Interactive options (fallback)
    p.add_argument("--interactive", action="store_true", help="Launch manual mask painter UI instead of automation.")
    p.add_argument("--brush", type=int, default=10, help="Initial brush size in pixels (interactive mode).")
    return p.parse_args()


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - entry point
    args = parse_args()
    # Ensure raw input dir exists to guide the user
    args.input_dir.mkdir(parents=True, exist_ok=True)

    if args.interactive or not args.checkpoint:
        # Launch interactive painter when requested, or when no checkpoint provided
        painter = MaskPainter(
            input_dir=args.input_dir,
            output_dataset=args.output_dataset,
            num_classes=args.num_classes,
            class_names=args.class_names,
            brush_size=args.brush,
        )
        painter.run()
        return

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
