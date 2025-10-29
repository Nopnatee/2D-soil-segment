from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from soil_segment.custom_unet import SimpleUNet
except ModuleNotFoundError:
    # Support running directly from the repository without `pip install -e .`
    here = Path(__file__).resolve().parents[1]
    src_path = here / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from soil_segment.custom_unet import SimpleUNet  # type: ignore

DEFAULT_PELLET_CLASS_NAMES: Tuple[str, ...] = (
    "background",
    "Black_DAP",
    "Red_MOP",
    "White_AMP",
    "White_Boron",
    "White_Mg",
    "Yellow_Urea",
)

@dataclass(frozen=True)
class ClassStat:
    """Container for per-class pellet statistics."""

    idx: int
    name: str
    pixel_count: int
    percentage: float


def _default_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "checkpoints" / "best_model.pth"


def _load_checkpoint_state(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(ckpt_path), map_location=device)

    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return payload["model_state_dict"]
        if "state_dict" in payload:
            return payload["state_dict"]
        # Some checkpoints might already be pure state dicts
        return payload

    # Support serialized torch.nn.Module objects
    if hasattr(payload, "state_dict"):
        return payload.state_dict()

    raise ValueError(f"Unsupported checkpoint format: {type(payload)}")


def _infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith("final_conv.weight"):
            return value.shape[0]
    raise KeyError("Could not infer number of classes from state dict.")


def _resolve_class_names(
    n_classes: int,
    cli_names: Optional[Sequence[str]],
    class_json: Optional[Path],
) -> List[str]:
    if cli_names:
        if len(cli_names) != n_classes:
            raise ValueError(f"--class-names expects {n_classes} entries, got {len(cli_names)}")
        return list(cli_names)

    if class_json and class_json.exists():
        try:
            data = json.loads(class_json.read_text())
            names = data.get("class_names")
            if isinstance(names, list) and len(names) >= n_classes:
                return names[:n_classes]
        except json.JSONDecodeError:
            pass  # Fall back to defaults if classes.json is malformed

    # Defaults aligned with the current fertilizer pellet taxonomy
    defaults: List[str] = []
    for idx in range(n_classes):
        if idx < len(DEFAULT_PELLET_CLASS_NAMES):
            defaults.append(DEFAULT_PELLET_CLASS_NAMES[idx])
        else:
            defaults.append(f"Pellet Class {idx}")
    return defaults


def _prepare_image_tensor(image_bgr: np.ndarray, img_size: int, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, (h, w)


def _default_palette(n: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [220, 20, 60],
            [0, 128, 0],
            [30, 144, 255],
            [255, 140, 0],
            [138, 43, 226],
            [255, 215, 0],
            [0, 206, 209],
            [205, 92, 92],
            [154, 205, 50],
        ],
        dtype=np.uint8,
    )
    if n <= base.shape[0]:
        return base[:n]

    rng = np.random.default_rng(12345)
    extra = rng.integers(0, 256, size=(n - base.shape[0], 3), dtype=np.uint8)
    return np.vstack([base, extra])


def _render_overlay(image_bgr: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color_mask = np.zeros_like(image_bgr, dtype=np.uint8)
    for class_id in np.unique(mask):
        if class_id >= len(palette):
            continue
        color_mask[mask == class_id] = palette[class_id]
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0)


def _collect_stats(mask: np.ndarray, class_names: Sequence[str]) -> List[ClassStat]:
    total_pixels = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))

    stats: List[ClassStat] = []
    for idx, name in enumerate(class_names):
        pixel_count = int(count_map.get(idx, 0))
        percentage = (pixel_count / total_pixels) * 100 if total_pixels else 0.0
        stats.append(ClassStat(idx=idx, name=name, pixel_count=pixel_count, percentage=percentage))
    return stats


# -----------------------------------------------------------------------------
# Core prediction routine
# -----------------------------------------------------------------------------


def predict_pellet_amounts(
    image_path: Path,
    checkpoint: Path,
    class_names: Sequence[str],
    img_size: int,
    device: torch.device,
    save_mask: Optional[Path] = None,
    save_overlay: Optional[Path] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> List[ClassStat]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to load image as BGR: {image_path}")

    if state_dict is None:
        state_dict = _load_checkpoint_state(checkpoint, device)

    n_classes = _infer_num_classes(state_dict)

    if len(class_names) != n_classes:
        raise ValueError(f"Model predicts {n_classes} classes but {len(class_names)} class names were provided.")

    model = SimpleUNet(n_classes=n_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tensor, (orig_h, orig_w) = _prepare_image_tensor(image_bgr, img_size, device)

    with torch.no_grad():
        logits = model(tensor)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long().squeeze(0).squeeze(0)
        else:
            pred = torch.argmax(logits, dim=1).squeeze(0)

    mask = pred.cpu().numpy().astype(np.uint8)
    if (orig_h, orig_w) != (img_size, img_size):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    stats = _collect_stats(mask, class_names)

    if save_mask:
        save_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_mask), mask)

    if save_overlay:
        palette = _default_palette(len(class_names))
        overlay = _render_overlay(image_bgr, mask, palette)
        save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_overlay), overlay)

    return stats


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fertilizer pellet amounts from an image.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the segmentation checkpoint (defaults to checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        help="Optional list of class names matching the model outputs (background first).",
    )
    parser.add_argument(
        "--class-json",
        type=Path,
        help="Optional path to a classes.json file containing `class_names`.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Square inference size expected by the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device to use (e.g., cuda, cpu). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--save-mask",
        type=Path,
        help="Optional path to save the raw predicted mask (PNG).",
    )
    parser.add_argument(
        "--save-overlay",
        type=Path,
        help="Optional path to save an overlay visualization (PNG).",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Suppress background class in the printed summary.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    checkpoint = args.checkpoint or _default_checkpoint_path()
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    state_dict = _load_checkpoint_state(checkpoint, device)
    num_classes = _infer_num_classes(state_dict)
    class_names = _resolve_class_names(
        n_classes=num_classes,
        cli_names=args.class_names,
        class_json=args.class_json,
    )

    stats = predict_pellet_amounts(
        image_path=args.image,
        checkpoint=checkpoint,
        class_names=class_names,
        img_size=args.img_size,
        device=device,
        save_mask=args.save_mask,
        save_overlay=args.save_overlay,
        state_dict=state_dict,
    )

    total_pellet_pixels = sum(stat.pixel_count for stat in stats if stat.idx != 0)

    print(f"\nFertilizer pellet analysis for {args.image}:")
    for stat in stats:
        if args.no_background and stat.idx == 0:
            continue
        prefix = "Pellet" if stat.idx != 0 else "Background"
        print(
            f"  [{stat.idx}] {stat.name}: {stat.pixel_count} px "
            f"({stat.percentage:.2f}% of image)"
        )
    print(f"\nTotal pellet pixels (excluding background): {total_pellet_pixels}")


if __name__ == "__main__": 
    main()
