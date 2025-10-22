import argparse
import os
import re
from typing import Dict, Any, Tuple, Optional

import torch
import matplotlib.pyplot as plt


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
    parser = argparse.ArgumentParser(description="Visualize training progress from trainer checkpoints.")
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
        help="Do not display the plot window (useful on headless machines).",
    )

    args = parser.parse_args()

    ckpt_path = _find_latest_checkpoint(args.path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not resolve a checkpoint from '{args.path}'. Provide a .pth file or a directory with checkpoints."
        )

    history, meta = load_history(ckpt_path)
    title = f"{os.path.basename(meta['path'])}"
    if meta.get("best_dice") is not None:
        title += f" | Best Dice: {meta['best_dice']:.4f}"

    # Default save path if requested without a filename
    save_path = args.save
    if save_path is None and os.path.isdir(args.path):
        save_path = os.path.join(args.path, "training_curves.png")

    plot_history(history, title=title, save=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
