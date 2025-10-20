"""Training utilities for the SimpleUNet soil segmentation model."""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from ..models.unet import SimpleUNet

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "assets" / "checkpoints"


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


class JointTransform:
    """Apply the same spatial augmentation to image and mask."""

    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Random rotation
        angle = random.uniform(-10, 10)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        # Random color jitter
        if random.random() > 0.5:
            image = self.color_jitter(image)

        # Convert to tensor and normalize image
        image = self.to_tensor(image)
        image = self.normalize(image)

        # Convert mask to long tensor
        mask_tensor = torch.from_numpy(np.array(mask, copy=True)).long()

        return image, mask_tensor


class BeadDataset(Dataset):
    """Dataset that returns (image, mask) pairs with optional joint transforms."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        joint_transform=None,
        debug: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.joint_transform = joint_transform
        self.debug = debug
        self.images = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        if self.debug:
            print(f"[BeadDataset] Found {len(self.images)} images in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = self.image_dir / img_name

        # Try matching extensions for the mask
        mask_path = self.mask_dir / img_name
        if not mask_path.exists():
            base_name = img_path.stem
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = self.mask_dir / f"{base_name}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            else:
                raise FileNotFoundError(f"No mask found for image {img_name} in {self.mask_dir}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_array = np.array(mask)
        if mask_array.ndim == 3:
            mask = Image.fromarray(mask_array[:, :, 0].astype(np.uint8), mode="L")

        if self.joint_transform:
            image_tensor, mask_tensor = self.joint_transform(image, mask)
        else:
            image_tensor = transforms.ToTensor()(image)
            image_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )(image_tensor)
            mask_tensor = torch.from_numpy(np.array(mask, copy=True)).long()

        return image_tensor, mask_tensor


class DiceScore(nn.Module):
    """Dice coefficient for segmentation evaluation."""

    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        dice_scores: list[torch.Tensor] = []

        for class_idx in range(self.num_classes):
            pred_i = pred[:, class_idx]
            target_i = (target == class_idx).float()

            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return torch.stack(dice_scores).mean()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data_dir: Path
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    num_classes: int = 5
    img_size: int = 256
    batch_size: int = 8
    epochs: int = 200
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: Optional[str] = None
    debug: bool = False


class SegmentationTrainer:
    """High-level training loop for the SimpleUNet model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        n_classes: int,
        lr: float,
        weight_decay: float,
        debug: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        self.debug = debug

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.dice_metric = DiceScore(n_classes)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "lr": [],
        }
        self.best_dice = 0.0
        self.best_model_state: Optional[dict] = None
        self.previous_lr = self.optimizer.param_groups[0]["lr"]

    def _debug_batch(self, data: torch.Tensor, target: torch.Tensor, output: torch.Tensor, phase: str) -> None:
        print(f"\n=== {phase.upper()} BATCH DEBUG ===")
        print(f"Input shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Target unique values: {torch.unique(target)}")
        pred_classes = torch.argmax(output, dim=1)
        print(f"Predicted classes unique: {torch.unique(pred_classes)}")
        print("=" * 40)

    def _step(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(data)
        loss = self.criterion(output, target)
        dice = self.dice_metric(output, target)
        return loss, dice, output

    def _run_epoch(self, train: bool) -> Tuple[float, float]:
        loader = self.train_loader if train else self.val_loader
        total_loss = 0.0
        total_dice = 0.0

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            for batch_idx, (data, target) in enumerate(loader):
                loss, dice, output = self._step(data, target)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.debug and batch_idx == 0:
                    self._debug_batch(data, target, output, phase="train" if train else "val")

                total_loss += loss.item()
                total_dice += dice.item()

        return total_loss / len(loader), total_dice / len(loader)

    def fit(self, epochs: int, checkpoint_dir: Path, print_every: int = 1) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        print("Training configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU memory: {total_mem:.1f} GB")
        print("-" * 60)

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_dice = self._run_epoch(train=True)
            val_loss, val_dice = self._run_epoch(train=False)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_reduced = current_lr != self.previous_lr
            if lr_reduced:
                self.previous_lr = current_lr

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice"].append(train_dice)
            self.history["val_dice"].append(val_dice)
            self.history["lr"].append(current_lr)

            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_dice": self.best_dice,
                        "history": self.history,
                    },
                    checkpoint_dir / "best_model.pth",
                )

            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - epoch_start
                status = ""
                if is_best:
                    status += " [BEST]"
                if lr_reduced:
                    status += f" [LR {current_lr:.6f}]"
                print(
                    f"Epoch {epoch + 1:4d}/{epochs} | "
                    f"Train L {train_loss:.4f} D {train_dice:.4f} | "
                    f"Val L {val_loss:.4f} D {val_dice:.4f} | "
                    f"{elapsed:.1f}s{status}"
                )

        total_minutes = (time.time() - start_time) / 60
        print("-" * 60)
        print(f"Training completed in {total_minutes:.1f} minutes")
        print(f"Best validation Dice: {self.best_dice:.4f}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best validation checkpoint into model.")

    # ------------------------------------------------------------------ #
    # Optional analysis helpers
    # ------------------------------------------------------------------ #

    def plot_history(self) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(self.history["train_loss"], label="Train")
        axes[0, 0].plot(self.history["val_loss"], label="Validation")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.history["train_dice"], label="Train")
        axes[0, 1].plot(self.history["val_dice"], label="Validation")
        axes[0, 1].set_title("Dice score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.history["lr"])
        axes[1, 0].set_title("Learning rate")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.history["val_loss"], label="Val loss", alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.history["val_dice"], label="Val Dice", color="orange", alpha=0.7)
        axes[1, 1].set_title("Validation metrics")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate(self, loader: DataLoader) -> Tuple[Sequence[int], Sequence[int]]:
        self.model.eval()
        predictions: list[int] = []
        targets: list[int] = []

        with torch.no_grad():
            for data, target in tqdm(loader, desc="Evaluating"):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = self.model(data)
                pred = torch.argmax(output, dim=1)

                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())

        print("\nClassification report:")
        print(classification_report(targets, predictions))

        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return predictions, targets


# ---------------------------------------------------------------------------
# High level orchestration
# ---------------------------------------------------------------------------


def create_data_loaders(
    data_dir: Path,
    *,
    img_size: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 42,
    debug: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected 'images' and 'masks' folders under {data_dir}")

    random.seed(random_seed)

    augmentation = JointTransform(img_size=img_size)

    def val_test_transform(image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(image_tensor)
        mask_tensor = torch.from_numpy(np.array(mask, copy=True)).long()
        return image_tensor, mask_tensor

    base_dataset = BeadDataset(
        image_dir,
        mask_dir,
        joint_transform=None,
        debug=debug,
    )

    total_samples = len(base_dataset)
    indices = list(range(total_samples))
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=random_seed,
        shuffle=True,
    )

    train_dataset = BeadDataset(image_dir, mask_dir, joint_transform=augmentation, debug=debug)
    val_dataset = BeadDataset(image_dir, mask_dir, joint_transform=val_test_transform, debug=False)
    test_dataset = BeadDataset(image_dir, mask_dir, joint_transform=val_test_transform, debug=False)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Dataset split: total={total_samples} | "
        f"train={len(train_indices)} | val={len(val_indices)} | test={len(test_indices)}"
    )

    return train_loader, val_loader, test_loader


def inspect_masks(dataset: Dataset, max_samples: int = 3) -> None:
    print("\nMask statistics preview:")
    for i in range(min(max_samples, len(dataset))):
        _, mask = dataset[i]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        unique = np.unique(mask_np)
        counts = np.bincount(mask_np.flatten(), minlength=int(unique.max()) + 1 if unique.size else 0)
        print(f"Sample {i}: shape={mask_np.shape}, classes={unique}")
        for class_idx in unique:
            count = counts[class_idx] if class_idx < len(counts) else 0
            pct = (count / mask_np.size) * 100
            print(f"  class {class_idx}: {count} px ({pct:.1f}%)")
    print("-" * 40)


def visualize_prediction(model: nn.Module, sample: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> None:
    model.eval()
    image, mask = sample
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.squeeze().permute(1, 2, 0).cpu())
    axes[0].set_title("Image")
    axes[1].imshow(mask.cpu() if isinstance(mask, torch.Tensor) else mask)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def train_unet(config: TrainingConfig) -> SegmentationTrainer:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = create_data_loaders(
        config.data_dir,
        img_size=config.img_size,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        debug=config.debug,
    )

    inspect_masks(train_loader.dataset.dataset, max_samples=3)  # type: ignore[arg-type]

    model = SimpleUNet(n_classes=config.num_classes)
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=config.num_classes,
        lr=config.lr,
        weight_decay=config.weight_decay,
        debug=config.debug,
    )

    trainer.fit(config.epochs, config.checkpoint_dir)
    trainer.evaluate(test_loader)

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the SimpleUNet segmentation model.")
    parser.add_argument("--data-dir", type=Path, default=Path("UNET_dataset"), help="Dataset root containing images/ and masks/")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Directory to store checkpoints")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--img-size", type=int, default=256, help="Square image size expected by the network")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Portion of dataset for training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Portion of dataset for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Portion of dataset for testing")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debugging output")

    args = parser.parse_args(argv)

    return TrainingConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_classes=args.num_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        debug=args.debug,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    train_unet(config)


if __name__ == "__main__":
    main()
