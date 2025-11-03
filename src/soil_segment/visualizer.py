import argparse
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Sequence

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import patches as mpatches
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Support running as a script or within the package
if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from soil_segment.custom_unet import SimpleUNet
else:
    from .custom_unet import SimpleUNet


def _find_latest_checkpoint(path: str) -> Optional[str]:
    """Resolve a checkpoint file.

    - If `path` is a file, return it.
    - If `path` is a directory, prefer `best_model.pth`, otherwise pick the
      highest `checkpoint_epoch_*.pth`.
    """
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        return None

    best_path = os.path.join(path, "best_model.pth")
    if os.path.isfile(best_path):
        return best_path

    # Fallback: latest numbered checkpoint
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    latest: Tuple[int, Optional[str]] = (-1, None)
    for fname in os.listdir(path):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            if epoch > latest[0]:
                latest = (epoch, os.path.join(path, fname))

    return latest[1]


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CLASS_NAMES = (
    "background",
    "Black_DAP",
    "Red_MOP",
    "White_AMP",
    "White_Boron",
    "White_Mg",
    "Yellow_Urea",
)
DEFAULT_CLASS_COLORS = [
    (0, 0, 0),        # background - black
    (220, 20, 60),    # Black_DAP - crimson
    (178, 34, 34),    # Red_MOP - firebrick
    (135, 206, 250),  # White_AMP - light sky blue
    (255, 182, 193),  # White_Boron - light pink
    (144, 238, 144),  # White_Mg - light green
    (255, 215, 0),    # Yellow_Urea - gold
]


class PredictionDataset(Dataset):
    """Dataset wrapper that mirrors validation preprocessing."""

    def __init__(self, data_dir: str, img_size: int = 1024):
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.img_size = img_size

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.images: List[str] = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if not self.images:
            raise ValueError(f"No image files found in {self.image_dir}")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.images)

    def _resolve_mask(self, image_name: str) -> str:
        mask_path = os.path.join(self.mask_dir, image_name)
        if os.path.exists(mask_path):
            return mask_path

        base = os.path.splitext(image_name)[0]
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Mask not found for image '{image_name}' in {self.mask_dir}")

    def __getitem__(self, idx: int):
        file_name = self.images[idx]
        image_path = os.path.join(self.image_dir, file_name)
        mask_path = self._resolve_mask(file_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image_tensor = self.normalize(self.to_tensor(image))
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image_tensor, mask_tensor, file_name


def _denormalize_tensor(image: torch.Tensor) -> torch.Tensor:
    """Bring a normalized CHW tensor back to [0, 1] range."""
    mean = torch.tensor(_IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def predict_img(
    checkpoint_path: str,
    dataset_dir: str,
    *,
    img_size: int = 1024,
    n_classes: int = 7,
    device: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """Run model predictions across a dataset and provide a scrollable viewer."""
    resolved_ckpt = _find_latest_checkpoint(checkpoint_path)
    if resolved_ckpt is None:
        raise FileNotFoundError(f"Could not resolve checkpoint from '{checkpoint_path}'")

    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SimpleUNet(n_classes=n_classes).to(device_obj)

    checkpoint = torch.load(resolved_ckpt, map_location=device_obj)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = PredictionDataset(dataset_dir, img_size=img_size)

    labels = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
    if len(labels) < n_classes:
        labels.extend(f"class_{idx}" for idx in range(len(labels), n_classes))
    labels = labels[:n_classes]

    base_colors = np.array(DEFAULT_CLASS_COLORS, dtype=np.float32) / 255.0
    if base_colors.shape[0] < n_classes:
        cmap_extra = plt.get_cmap("tab20")
        extra_needed = n_classes - base_colors.shape[0]
        extra_colors = cmap_extra(np.linspace(0, 1, extra_needed))[:, :3]
        base_colors = np.vstack([base_colors, extra_colors])
    color_palette = base_colors[:n_classes]
    listed_cmap = ListedColormap(color_palette)
    norm = BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)

    samples = []
    for idx in range(len(dataset)):
        image_tensor, mask_tensor, file_name = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device_obj)

        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        image_vis = _denormalize_tensor(image_tensor.cpu()).permute(1, 2, 0).numpy()
        mask_vis = mask_tensor.cpu().numpy()
        samples.append(
            {
                "image": image_vis,
                "mask": mask_vis,
                "prediction": prediction,
                "name": file_name,
            }
        )

    if not samples:
        raise ValueError("No samples available for visualization.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(top=0.82, bottom=0.2 if len(samples) > 1 else 0.1)

    legend_ax = fig.add_axes([0.05, 0.85, 0.9, 0.08])
    legend_ax.axis("off")

    def _draw_legend() -> None:
        legend_ax.cla()
        legend_ax.axis("off")
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        total = len(labels)
        if total == 0:
            return
        padding = 0.02
        width = (1 - 2 * padding) / total
        box_height = 0.3
        y_box = 0.25
        for idx, (label, color) in enumerate(zip(labels, color_palette)):
            x_pos = padding + idx * width
            legend_ax.add_patch(
                mpatches.Rectangle(
                    (x_pos, y_box),
                    width * 0.6,
                    box_height,
                    color=color,
                    transform=legend_ax.transAxes,
                    clip_on=False,
                )
            )
            legend_ax.text(
                x_pos + width * 0.3,
                y_box + box_height + 0.05,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                transform=legend_ax.transAxes,
            )
        legend_ax.figure.canvas.draw_idle()

    _draw_legend()

    def _render_sample(sample_idx: int) -> None:
        sample = samples[sample_idx]
        axes[0].cla()
        axes[1].cla()
        axes[2].cla()

        axes[0].imshow(sample["image"])
        axes[0].set_title(f"Image: {sample['name']}")

        axes[1].imshow(sample["mask"], cmap=listed_cmap, norm=norm, interpolation="nearest")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(sample["prediction"], cmap=listed_cmap, norm=norm, interpolation="nearest")
        axes[2].set_title("Prediction")

        for ax in axes:
            ax.axis("off")

        fig.canvas.draw_idle()

    _render_sample(0)

    slider = None
    if len(samples) > 1:
        slider_ax = fig.add_axes([0.1, 0.08, 0.8, 0.04])
        slider = Slider(
            slider_ax,
            "Sample",
            valmin=0,
            valmax=len(samples) - 1,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )

        def _on_change(val: float) -> None:
            idx = int(np.clip(val, 0, len(samples) - 1))
            _render_sample(idx)

        slider.on_changed(_on_change)

        def _on_key(event) -> None:
            if slider is None:
                return
            current = int(slider.val)
            if event.key in ("right", "down"):
                slider.set_val(min(len(samples) - 1, current + 1))
            elif event.key in ("left", "up"):
                slider.set_val(max(0, current - 1))

        fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show()


def load_history(ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load `history` and meta info from a trainer checkpoint.

    Returns (history, meta). `history` follows trainer.py keys:
    - train_loss, val_loss, train_dice, val_dice, lr
    Meta may include: epoch, best_dice.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history", {})

    # Ensure required keys exist with empty lists if missing
    for k in ["train_loss", "val_loss", "train_dice", "val_dice", "lr"]:
        history.setdefault(k, [])

    meta = {
        "epoch": ckpt.get("epoch"),
        "best_dice": ckpt.get("best_dice"),
        "path": ckpt_path,
    }
    return history, meta


def plot_history(history: Dict[str, Any], title: Optional[str] = None, save: Optional[str] = None, show: bool = True) -> None:
    """Plot training curves to screen and/or file."""
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_dice = history.get("train_dice", [])
    val_dice = history.get("val_dice", [])
    lr = history.get("lr", [])

    n_epochs = max(len(train_loss), len(val_loss), len(train_dice), len(val_dice), len(lr))
    if n_epochs == 0:
        raise ValueError("No history found to plot. Train the model or select another checkpoint.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    if title:
        fig.suptitle(title, fontsize=14)

    # Loss curves
    axes[0, 0].plot(train_loss, label="Train Loss")
    axes[0, 0].plot(val_loss, label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Dice curves
    axes[0, 1].plot(train_dice, label="Train Dice")
    axes[0, 1].plot(val_dice, label="Val Dice")
    axes[0, 1].set_title("Dice Score")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate
    axes[1, 0].plot(lr)
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    # Avoid log of non-positive if the schedule ever reaches 0
    if all(x > 0 for x in lr):
        axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True)

    # Validation overlay
    axes[1, 1].plot(val_loss, label="Val Loss", color="tab:blue")
    ax2 = axes[1, 1].twinx()
    ax2.plot(val_dice, label="Val Dice", color="tab:orange")
    axes[1, 1].set_title("Validation Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss", color="tab:blue")
    ax2.set_ylabel("Dice", color="tab:orange")
    axes[1, 1].grid(True)

    # Handle legends for twin axes
    lines, labels = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout(rect=(0, 0, 1, 0.96) if title else None)

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress and predictions from checkpoints.")
    parser.add_argument(
        "path",
        nargs="?",
        default="checkpoints",
        help="Path to a checkpoint file (.pth) or a directory containing checkpoints. Defaults to 'checkpoints'.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the plotted figure (e.g., checkpoints/training_curves.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the history plot window (useful on headless machines).",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Disable the interactive prediction viewer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset root containing 'images' and 'masks' subdirectories (required with --predict).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="Resize dimension used for prediction visualization (defaults to 1024).",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=7,
        help="Number of segmentation classes for the model (defaults to 7).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier for inference, e.g. 'cpu' or 'cuda:0'. Defaults to auto selection.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip plotting the training history (useful when only the prediction viewer is needed).",
    )

    args = parser.parse_args()

    ckpt_path = _find_latest_checkpoint(args.path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not resolve a checkpoint from '{args.path}'. Provide a .pth file or a directory with checkpoints."
        )

    if not args.no_history:
        history, meta = load_history(ckpt_path)
        title = f"{os.path.basename(meta['path'])}"
        if meta.get("best_dice") is not None:
            title += f" | Best Dice: {meta['best_dice']:.4f}"

        # Default save path if requested without a filename
        save_path = args.save
        if save_path is None and os.path.isdir(args.path):
            save_path = os.path.join(args.path, "training_curves.png")

        plot_history(history, title=title, save=save_path, show=not args.no_show)

    if not args.no_predict:
        dataset_path = args.dataset
        if dataset_path is None:
            default_dataset = Path(__file__).resolve().parents[2] / "datasets" / "UNET_dataset"
            if default_dataset.is_dir():
                dataset_path = str(default_dataset)
        if dataset_path is None:
            raise ValueError("Dataset path is required for predictions. Provide --dataset or use --no-predict.")
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        predict_img(
            checkpoint_path=ckpt_path,
            dataset_dir=dataset_path,
            img_size=args.img_size,
            n_classes=args.n_classes,
            device=args.device,
            class_names=DEFAULT_CLASS_NAMES,
        )

if __name__ == "__main__":
    main()
