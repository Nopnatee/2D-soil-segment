import argparse
import sys
from pathlib import Path
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import random
from typing import Optional

CLASS_NAMES = (
    'background',
    'Black_DAP',
    'Red_MOP',
    'White_AMP',
    'White_Boron',
    'White_Mg',
    'Yellow_Urea_coated',
    'Yellow_Urea_uncoated',
)
N_CLASSES = len(CLASS_NAMES)
MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def get_class_names(n_classes):
    """Return class names trimmed/padded to match n_classes."""
    if n_classes <= len(CLASS_NAMES):
        return list(CLASS_NAMES[:n_classes])
    return list(CLASS_NAMES) + [
        f'Class {i}' for i in range(len(CLASS_NAMES), n_classes)
    ]


def append_name_suffix(path: Path, suffix: str) -> Path:
    """Append a suffix to the final path segment name."""
    if not suffix:
        return path
    return path.with_name(f"{path.name}{suffix}")


def resolve_mask_path(mask_dir, img_name):
    """Resolve a mask path for an image, including *_mask exports."""
    base_name = os.path.splitext(img_name)[0]
    candidates = [img_name]
    candidates.extend(
        base_name + ext for ext in MASK_EXTENSIONS if base_name + ext != img_name
    )
    candidates.extend(base_name + '_mask' + ext for ext in MASK_EXTENSIONS)

    for candidate in candidates:
        mask_path = os.path.join(mask_dir, candidate)
        if os.path.exists(mask_path):
            return mask_path

    raise FileNotFoundError(
        f"No mask found for image {img_name} in {mask_dir}"
    )

# Support running as a script or within the package
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from soil_segment.custom_unet import SimpleUNet, ConvBlock
    from soil_segment.config import get_data_paths
else:
    from .custom_unet import SimpleUNet, ConvBlock
    from .config import get_data_paths


# ============================================================================
# JOINT TRANSFORM CLASS - Applies same transform to image and mask
# ============================================================================
class JointTransform:
    """Applies synchronized transforms to both image and mask"""
    def __init__(self, img_size=1024, is_training=True):
        self.img_size = img_size
        self.is_training = is_training
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if is_training:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            )

    def __call__(self, image, mask):
        # Always resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Training augmentations only
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random vertical flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # Random rotation (±15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

            # Random color jitter
            if random.random() > 0.5:
                image = self.color_jitter(image)

            # Random Gaussian blur
            if random.random() > 0.3:
                from PIL import ImageFilter
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))

        # Convert to tensor
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Convert mask to long tensor
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


# ============================================================================
# DATASET CLASS - Single instance with transform parameter
# ============================================================================
class BeadDataset(Dataset):
    """Dataset for NPK bead segmentation with flexible transforms"""
    def __init__(self, image_dir, mask_dir, transform=None, debug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.debug = debug
        
        # Sort for deterministic ordering
        self.images = sorted(
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

        if self.debug:
            print(f"[BeadDataset] Found {len(self.images)} images in {image_dir}")
            print(f"[BeadDataset] First 3 images: {self.images[:3]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = resolve_mask_path(self.mask_dir, img_name)

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Convert mask to single channel if needed
        mask_array = np.array(mask)
        if len(mask_array.shape) > 2:
            mask_array = mask_array[:, :, 0]
        mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')

        # Debug first few samples
        if self.debug and idx < 3:
            print(f"\n[BeadDataset] Sample {idx}: {img_name}")
            print(f"  Image: {img_path}")
            print(f"  Mask: {mask_path}")
            print(f"  Mask unique values: {np.unique(mask_array)}")

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


# ============================================================================
# DATASET WRAPPER - Applies different transforms to subsets
# ============================================================================
class TransformSubset(Dataset):
    """Wrapper to apply specific transform to a subset of data"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, mask = load_image_mask_pair(self.dataset, original_idx)
        
        # Apply this subset's specific transform
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask


def load_image_mask_pair(dataset, original_idx):
    """Load a raw image/mask pair from the base dataset by absolute index."""
    img_name = dataset.images[original_idx]
    img_path = os.path.join(dataset.image_dir, img_name)
    mask_path = resolve_mask_path(dataset.mask_dir, img_name)

    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path)

    mask_array = np.array(mask)
    if len(mask_array.shape) > 2:
        mask_array = mask_array[:, :, 0]
    mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')

    return image, mask


class RandomPatchDataset(Dataset):
    """Patch-based dataset with stronger augmentation for tiny-data training."""

    def __init__(self, dataset, indices, patch_size=512, patches_per_image=64):
        self.dataset = dataset
        self.indices = list(indices)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03,
        )

    def __len__(self):
        return len(self.indices) * self.patches_per_image

    def _apply_strong_augmentations(self, image, mask):
        # Flips
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Mild affine geometry perturbation
        if random.random() < 0.8:
            angle, translations, scale, shear = transforms.RandomAffine.get_params(
                degrees=(-20, 20),
                translate=(0.08, 0.08),
                scale_ranges=(0.9, 1.1),
                shears=(-5, 5),
                img_size=image.size,
            )
            image = TF.affine(
                image,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0,
            )

        if random.random() < 0.7:
            image = self.color_jitter(image)

        if random.random() < 0.35:
            from PIL import ImageFilter
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2))
            )

        return image, mask

    def _random_crop(self, image, mask):
        width, height = image.size

        if width < self.patch_size or height < self.patch_size:
            resize_w = max(width, self.patch_size)
            resize_h = max(height, self.patch_size)
            image = image.resize((resize_w, resize_h), Image.BILINEAR)
            mask = mask.resize((resize_w, resize_h), Image.NEAREST)
            width, height = image.size

        max_left = width - self.patch_size
        max_top = height - self.patch_size
        left = 0 if max_left <= 0 else random.randint(0, max_left)
        top = 0 if max_top <= 0 else random.randint(0, max_top)

        image = TF.crop(image, top, left, self.patch_size, self.patch_size)
        mask = TF.crop(mask, top, left, self.patch_size, self.patch_size)
        return image, mask

    def __getitem__(self, idx):
        original_idx = self.indices[idx % len(self.indices)]
        image, mask = load_image_mask_pair(self.dataset, original_idx)

        image, mask = self._apply_strong_augmentations(image, mask)
        image, mask = self._random_crop(image, mask)

        image = self.to_tensor(image)
        if random.random() < 0.3:
            image = (image + torch.randn_like(image) * random.uniform(0.01, 0.03)).clamp(0, 1)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ============================================================================
# METRICS
# ============================================================================
class DiceScore(nn.Module):
    """Dice coefficient for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1e-6, ignore_empty_target=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_empty_target = ignore_empty_target
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] - logits
        target: [B, H, W] - class indices
        """
        pred = F.softmax(pred, dim=1)
        dice_scores = []
        
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = (target == i).float()

            if self.ignore_empty_target and target_i.sum() <= 0:
                continue
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        if not dice_scores:
            return pred.new_tensor(0.0)
        return torch.stack(dice_scores).mean()


class IoUScore(nn.Module):
    """Intersection over Union for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1e-6, ignore_empty_target=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_empty_target = ignore_empty_target
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        iou_scores = []
        for i in range(self.num_classes):
            pred_i = (pred == i)
            target_i = (target == i)

            if self.ignore_empty_target and target_i.float().sum() <= 0:
                continue
            
            intersection = (pred_i & target_i).float().sum()
            union = (pred_i | target_i).float().sum()
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou)

        if not iou_scores:
            return pred.new_tensor(0.0)
        return torch.stack(iou_scores).mean()


class WeightedCrossEntropyDiceLoss(nn.Module):
    """Weighted CE + Dice loss for imbalanced multi-class segmentation."""

    def __init__(
        self,
        num_classes,
        class_weights: Optional[torch.Tensor] = None,
        ce_ratio=0.6,
        dice_ratio=0.4,
        smooth=1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_ratio = float(ce_ratio)
        self.dice_ratio = float(dice_ratio)
        self.smooth = float(smooth)

        if class_weights is None:
            self.register_buffer("class_weights", torch.ones(self.num_classes))
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.class_weights)

        probs = F.softmax(logits, dim=1)
        target_1hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * target_1hot).sum(dim=dims)
        union = probs.sum(dim=dims) + target_1hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Weighted mean dice: classes absent from training weights contribute 0.
        weight_sum = self.class_weights.sum().clamp_min(1e-8)
        dice_mean = (dice * self.class_weights).sum() / weight_sum
        dice_loss = 1.0 - dice_mean

        return self.ce_ratio * ce + self.dice_ratio * dice_loss


# ============================================================================
# TRAINER CLASS
# ============================================================================
class SegmentationTrainer:
    """Training pipeline for UNet with proper handling of small datasets"""
    
    def __init__(self, model, device, train_loader, val_loader, n_classes=N_CLASSES, 
                 learning_rate=1e-3, weight_decay=1e-4, criterion=None,
                 ignore_empty_classes=False):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        
        # Loss and metrics
        self.criterion = criterion.to(device) if criterion is not None else nn.CrossEntropyLoss()
        self.dice_metric = DiceScore(n_classes, ignore_empty_target=ignore_empty_classes)
        self.iou_metric = IoUScore(n_classes, ignore_empty_target=ignore_empty_classes)
        
        # Optimizer with proper weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts (good for small datasets)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_dice = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for data, target in pbar:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate losses
            loss = self.criterion(output, target)
            dice = self.dice_metric(output, target)
            iou = self.iou_metric(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            running_dice += dice.item()
            running_iou += iou.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })
        
        avg_loss = running_loss / len(self.train_loader)
        avg_dice = running_dice / len(self.train_loader)
        avg_iou = running_iou / len(self.train_loader)
        
        return avg_loss, avg_dice, avg_iou
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                
                loss = self.criterion(output, target)
                dice = self.dice_metric(output, target)
                iou = self.iou_metric(output, target)
                
                running_loss += loss.item()
                running_dice += dice.item()
                running_iou += iou.item()
        
        avg_loss = running_loss / len(self.val_loader)
        avg_dice = running_dice / len(self.val_loader)
        avg_iou = running_iou / len(self.val_loader)
        
        return avg_loss, avg_dice, avg_iou
    
    def train(
        self,
        num_epochs,
        save_dir='checkpoints',
        early_stopping_patience=50,
        model_suffix='',
    ):
        """Full training loop with early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        best_model_name = f"best_model{model_suffix}.pth"
        
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(self.train_loader.dataset)}")
        print(f"  Validation samples: {len(self.val_loader.dataset)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Initial LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_dice, train_iou = self.train_epoch()
            
            # Validation phase
            val_loss, val_dice, val_iou = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['lr'].append(current_lr)
            
            # Check if best model
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.best_epoch = epoch
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_dice': self.best_dice,
                    'history': self.history
                }, os.path.join(save_dir, best_model_name))
            else:
                self.patience_counter += 1
            
            # Regular checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': self.history
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}{model_suffix}.pth'))
            
            # Print progress
            epoch_time = time.time() - epoch_start
            status = f"Epoch {epoch+1:3d}/{num_epochs}"
            status += f" | Train L:{train_loss:.4f} D:{train_dice:.4f} IoU:{train_iou:.4f}"
            status += f" | Val L:{val_loss:.4f} D:{val_dice:.4f} IoU:{val_iou:.4f}"
            status += f" | LR:{current_lr:.6f} | {epoch_time:.1f}s"
            
            if is_best:
                status += " [BEST]"
            
            print(status)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation Dice: {self.best_dice:.4f} at epoch {self.best_epoch + 1}")
                print(f"{'='*70}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation Dice: {self.best_dice:.4f} at epoch {self.best_epoch + 1}")
        print(f"{'='*70}\n")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Best model loaded for evaluation.")
    
    def plot_training_history(self, save_path=None):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(self.history['val_loss'], label='Val', alpha=0.7)
        axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice curves
        axes[0, 1].plot(self.history['train_dice'], label='Train', alpha=0.7)
        axes[0, 1].plot(self.history['val_dice'], label='Val', alpha=0.7)
        axes[0, 1].axhline(y=self.best_dice, color='r', linestyle='--', 
                          label=f'Best: {self.best_dice:.4f}', alpha=0.5)
        axes[0, 1].set_title('Dice Score', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU curves
        axes[0, 2].plot(self.history['train_iou'], label='Train', alpha=0.7)
        axes[0, 2].plot(self.history['val_iou'], label='Val', alpha=0.7)
        axes[0, 2].set_title('IoU Score', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(self.history['lr'], color='green', alpha=0.7)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting gap (Val - Train)
        dice_gap = np.array(self.history['val_dice']) - np.array(self.history['train_dice'])
        axes[1, 1].plot(dice_gap, color='purple', alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Generalization Gap (Val - Train Dice)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Dice Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined view
        ax = axes[1, 2]
        ax.plot(self.history['val_loss'], label='Val Loss', alpha=0.6, color='blue')
        ax.set_ylabel('Val Loss', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax.twinx()
        ax2.plot(self.history['val_dice'], label='Val Dice', alpha=0.6, color='orange')
        ax2.set_ylabel('Val Dice', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_title('Combined Validation Metrics', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_loader, class_names=None):
        """Comprehensive model evaluation"""
        if class_names is None:
            class_names = get_class_names(self.n_classes)
        else:
            class_names = list(class_names)
            if len(class_names) < self.n_classes:
                class_names.extend(
                    [f'Class {i}' for i in range(len(class_names), self.n_classes)]
                )
            class_names = class_names[:self.n_classes]
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        print("\nEvaluating model on test set...")
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                pred = torch.argmax(output, dim=1)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Classification report
        print("\n" + "="*70)
        print("Classification Report:")
        print("="*70)
        labels = list(range(self.n_classes))
        print(classification_report(
            all_targets, 
            all_predictions, 
            labels=labels,
            target_names=class_names,
            zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return all_predictions, all_targets


# ============================================================================
# DATA LOADER CREATION - FIXED VERSION
# ============================================================================
def create_data_loaders(data_dir, batch_size=2, img_size=1024,
                        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                        random_seed=42, num_workers=0):
    """
    Create train/val/test loaders with proper transform handling
    
    FIXED: No data leakage - uses single dataset with different transforms per split
    """
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    print("\n" + "="*70)
    print("Creating Data Loaders")
    print("="*70)
    
    # Create base dataset (no transforms yet)
    base_dataset = BeadDataset(
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'masks'),
        transform=None,
        debug=False
    )
    
    total_samples = len(base_dataset)
    print(f"Total samples found: {total_samples}")
    
    # Split indices
    indices = list(range(total_samples))
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_size,
        random_state=random_seed, 
        shuffle=True
    )
    val_indices, test_indices = train_test_split(
        temp_indices, 
        train_size=val_size,
        random_state=random_seed, 
        shuffle=True
    )
    
    # Create transforms
    train_transform = JointTransform(img_size=img_size, is_training=True)
    val_test_transform = JointTransform(img_size=img_size, is_training=False)
    
    # Create subsets with appropriate transforms
    train_dataset = TransformSubset(base_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(base_dataset, val_indices, val_test_transform)
    test_dataset = TransformSubset(base_dataset, test_indices, val_test_transform)
    
    # Log splits
    def log_split(name, indices):
        files = [base_dataset.images[int(i)] for i in indices]
        print(f"\n{name} Set ({len(files)} samples):")
        if len(files) <= 10:
            for f in files:
                print(f"  - {f}")
        else:
            for f in files[:5]:
                print(f"  - {f}")
            print(f"  ... and {len(files) - 5} more")
    
    log_split("Train", train_indices)
    log_split("Val", val_indices)
    log_split("Test", test_indices)
    
    # Create data loaders
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n{'='*70}")
    print(f"Data loaders created successfully")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader


def compute_class_weights_from_indices(dataset, indices, n_classes):
    """Compute inverse-frequency class weights from a subset of masks."""
    pixel_counts = np.zeros(int(n_classes), dtype=np.float64)

    for original_idx in indices:
        _, mask = load_image_mask_pair(dataset, original_idx)
        mask_np = np.array(mask).astype(np.int64)
        bincount = np.bincount(mask_np.flatten(), minlength=n_classes)
        pixel_counts += bincount[:n_classes]

    total_pixels = float(pixel_counts.sum())
    weights = np.zeros(int(n_classes), dtype=np.float32)
    present = pixel_counts > 0
    if total_pixels > 0 and present.any():
        weights[present] = total_pixels / (n_classes * pixel_counts[present])
        weights[present] = weights[present] / max(weights[present].mean(), 1e-8)
        weights[present] = np.clip(weights[present], 0.25, 6.0)
    else:
        weights[:] = 1.0

    return torch.tensor(weights, dtype=torch.float32), pixel_counts


def create_uncoated_fold_loaders(
    base_dataset,
    train_indices,
    val_indices,
    batch_size=4,
    patch_size=512,
    patches_per_image=64,
    num_workers=0,
):
    """Create patch-based train loader and deterministic full-image val loader."""
    train_dataset = RandomPatchDataset(
        base_dataset,
        train_indices,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
    )
    val_dataset = TransformSubset(
        base_dataset,
        val_indices,
        transform=JointTransform(img_size=patch_size, is_training=False),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def initialize_model_from_checkpoint(model, checkpoint_path):
    """Load compatible weights from a checkpoint (supports class-count mismatch)."""
    if checkpoint_path is None:
        return

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"[WARN] Init checkpoint not found, training from scratch: {checkpoint_path}")
        return

    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()

    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model.load_state_dict(compatible, strict=False)
    print(
        f"[Init] Loaded {len(compatible)}/{len(model_state)} tensors from "
        f"{checkpoint_path.name}"
    )
    if skipped:
        print(f"[Init] Skipped incompatible tensors: {len(skipped)}")


def run_uncoated_leave_one_out_cv(
    dataset_dir,
    checkpoints_dir,
    device,
    n_classes,
    class_names,
    model_suffix="_uncoated",
    patch_size=512,
    patches_per_image=64,
    batch_size=4,
    num_epochs=160,
    early_stopping_patience=30,
    learning_rate=1e-4,
    weight_decay=1e-4,
    random_seed=42,
    num_workers=0,
    init_checkpoint_path=None,
):
    """Run leave-one-out CV with patch-based training for uncoated data."""
    base_dataset = BeadDataset(
        os.path.join(dataset_dir, 'images'),
        os.path.join(dataset_dir, 'masks'),
        transform=None,
        debug=False,
    )
    total_samples = len(base_dataset)
    if total_samples < 2:
        raise RuntimeError(
            f"Uncoated LOO-CV requires at least 2 images, found {total_samples}"
        )

    print("\n" + "=" * 70)
    print("Uncoated Leave-One-Out Cross Validation")
    print("=" * 70)
    print(f"Total images: {total_samples}")
    print(f"Patch size: {patch_size}")
    print(f"Patches/image/epoch: {patches_per_image}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("=" * 70 + "\n")

    os.makedirs(checkpoints_dir, exist_ok=True)
    all_indices = list(range(total_samples))
    fold_results = []

    for fold_idx, val_index in enumerate(all_indices, start=1):
        fold_seed = random_seed + fold_idx
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        train_indices = [i for i in all_indices if i != val_index]
        val_indices = [val_index]
        val_image_name = base_dataset.images[val_index]

        print("\n" + "-" * 70)
        print(f"Fold {fold_idx}/{total_samples}")
        print(f"Validation image: {val_image_name}")
        print("-" * 70)

        train_loader, val_loader = create_uncoated_fold_loaders(
            base_dataset=base_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=batch_size,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            num_workers=num_workers,
        )

        class_weights, pixel_counts = compute_class_weights_from_indices(
            base_dataset, train_indices, n_classes
        )
        print(f"Class weights: {class_weights.tolist()}")
        print(f"Train pixel counts: {pixel_counts.astype(np.int64).tolist()}")

        criterion = WeightedCrossEntropyDiceLoss(
            num_classes=n_classes,
            class_weights=class_weights,
            ce_ratio=0.6,
            dice_ratio=0.4,
        )

        model = SimpleUNet(n_classes=n_classes)
        initialize_model_from_checkpoint(model, init_checkpoint_path)
        trainer = SegmentationTrainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=n_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            criterion=criterion,
            ignore_empty_classes=True,
        )

        fold_model_suffix = f"_fold{fold_idx}{model_suffix}"
        trainer.train(
            num_epochs=num_epochs,
            save_dir=checkpoints_dir,
            early_stopping_patience=early_stopping_patience,
            model_suffix=fold_model_suffix,
        )

        fold_result = {
            "fold": fold_idx,
            "val_image": val_image_name,
            "best_val_dice": float(trainer.best_dice),
            "best_epoch": int(trainer.best_epoch + 1),
            "class_weights": [float(x) for x in class_weights.tolist()],
            "train_pixel_counts": [int(x) for x in pixel_counts.astype(np.int64).tolist()],
        }
        fold_results.append(fold_result)

    best_dice_values = np.array([f["best_val_dice"] for f in fold_results], dtype=np.float64)
    summary = {
        "mode": "uncoated_leave_one_out_cv",
        "folds": total_samples,
        "mean_best_val_dice": float(best_dice_values.mean()),
        "std_best_val_dice": float(best_dice_values.std(ddof=0)),
        "min_best_val_dice": float(best_dice_values.min()),
        "max_best_val_dice": float(best_dice_values.max()),
        "fold_results": fold_results,
        "config": {
            "patch_size": int(patch_size),
            "patches_per_image": int(patches_per_image),
            "batch_size": int(batch_size),
            "epochs": int(num_epochs),
            "early_stopping_patience": int(early_stopping_patience),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "n_classes": int(n_classes),
            "class_names": list(class_names),
            "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else None,
        },
    }

    summary_path = Path(checkpoints_dir) / f"loo_cv_summary{model_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("LOO-CV Summary (Best Val Dice per fold)")
    for fold in fold_results:
        print(
            f"  Fold {fold['fold']}: "
            f"{fold['best_val_dice']:.4f} "
            f"(epoch {fold['best_epoch']}, val image={fold['val_image']})"
        )
    print("-" * 70)
    print(f"Mean Dice: {summary['mean_best_val_dice']:.4f}")
    print(f"Std Dice:  {summary['std_best_val_dice']:.4f}")
    print(f"Min/Max:   {summary['min_best_val_dice']:.4f} / {summary['max_best_val_dice']:.4f}")
    print(f"Saved summary: {summary_path}")
    print("=" * 70 + "\n")

    return summary


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_predictions(model, dataset, device, num_samples=4, 
                         class_names=None, checkpoint_path=None):
    """Visualize model predictions with ground truth"""
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model.eval()
    
    # Color map for visualization
    n_classes = model.final_conv.out_channels
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    if class_names is None:
        class_names = get_class_names(n_classes)
    
    # Randomly sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, mask = dataset[sample_idx]
            
            # Prepare image for model
            image_input = image.unsqueeze(0).to(device)
            
            # Denormalize image for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_display = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            
            # Get prediction
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            mask_np = mask.numpy()
            
            # Plot
            axes[idx, 0].imshow(image_display)
            axes[idx, 0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(mask_np, cmap=colors, vmin=0, vmax=n_classes-1)
            axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(pred, cmap=colors, vmin=0, vmax=n_classes-1)
            axes[idx, 2].set_title('Prediction', fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
    
    # Add colorbar legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors(i), label=class_names[i]) 
        for i in range(min(len(class_names), n_classes))
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=min(4, n_classes),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.show()


def verify_dataset_classes(data_dir, expected_classes=N_CLASSES):
    """Verify class labels in mask files and return sorted discovered classes."""
    print("\n" + "="*70)
    print("Verifying Dataset Classes")
    print("="*70)

    mask_dir = os.path.join(data_dir, 'masks')
    mask_files = sorted(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    all_classes = set()
    class_counts = {}

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        unique_classes = np.unique(mask)
        all_classes.update(unique_classes)

        for cls in unique_classes:
            class_counts[int(cls)] = class_counts.get(int(cls), 0) + 1

    found_classes = sorted(int(cls) for cls in all_classes)
    print(f"\nFound classes: {found_classes}")
    print(f"Expected classes: {list(range(expected_classes))}")

    if set(found_classes) == set(range(expected_classes)):
        print("[OK] All expected classes are present.")
    else:
        missing = set(range(expected_classes)) - set(found_classes)
        extra = set(found_classes) - set(range(expected_classes))
        if missing:
            print(f"[WARN] Missing classes: {sorted(missing)}")
        if extra:
            print(f"[WARN] Unexpected classes: {sorted(extra)}")

    print("\nClass distribution across dataset:")
    for cls in sorted(class_counts.keys()):
        print(f"  Class {cls}: appears in {class_counts[cls]} masks")

    print("="*70 + "\n")
    return found_classes


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train UNet segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to UNet dataset directory (expects images/ and masks/)",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="Directory to save training checkpoints and plots",
    )
    parser.add_argument(
        "--uncoated",
        action="store_true",
        help="Use dedicated *_uncoated dataset/checkpoint folders and model filenames",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--uncoated-patch-size",
        type=int,
        default=512,
        help="Patch size for uncoated patch-based training",
    )
    parser.add_argument(
        "--uncoated-patches-per-image",
        type=int,
        default=64,
        help="Number of random patches sampled per train image per epoch (uncoated mode)",
    )
    parser.add_argument(
        "--uncoated-batch-size",
        type=int,
        default=4,
        help="Batch size for uncoated mode",
    )
    parser.add_argument(
        "--uncoated-epochs",
        type=int,
        default=160,
        help="Epochs for each fold in uncoated leave-one-out CV",
    )
    parser.add_argument(
        "--uncoated-patience",
        type=int,
        default=30,
        help="Early stopping patience for uncoated leave-one-out CV",
    )
    parser.add_argument(
        "--uncoated-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for uncoated leave-one-out CV",
    )
    parser.add_argument(
        "--uncoated-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for uncoated leave-one-out CV",
    )
    parser.add_argument(
        "--uncoated-init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start each uncoated fold (defaults to checkpoints/best_model.pth if present)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main training pipeline"""
    args = parse_args(argv)

    suffix = "_uncoated" if args.uncoated else ""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*70}\n")
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Get paths
    paths = get_data_paths()
    default_dataset = append_name_suffix(Path(paths["unet_dataset"]), suffix)
    default_checkpoints = append_name_suffix(Path(paths["checkpoints"]), suffix)

    dataset_dir = str((args.dataset or default_dataset).expanduser().resolve())
    checkpoints_dir = str((args.checkpoints or default_checkpoints).expanduser().resolve())
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_model_name = f"best_model{suffix}.pth"
    curves_name = f"training_curves{suffix}.png"
    
    # Verify dataset first and align class count to actual labels
    found_classes = verify_dataset_classes(dataset_dir, expected_classes=N_CLASSES)
    if not found_classes:
        raise RuntimeError(f"No mask classes found under: {dataset_dir}")

    detected_n_classes = max(found_classes) + 1
    if detected_n_classes != N_CLASSES:
        print(
            f"[WARN] Configured n_classes={N_CLASSES}, "
            f"but dataset labels require n_classes={detected_n_classes}. "
            f"Using detected value."
        )
    active_n_classes = detected_n_classes
    class_names = get_class_names(active_n_classes)

    if args.uncoated:
        init_checkpoint_path = args.uncoated_init_checkpoint
        if init_checkpoint_path is None:
            default_init = Path(paths["checkpoints"]) / "best_model.pth"
            if default_init.exists():
                init_checkpoint_path = default_init
            else:
                print("[WARN] No default coated checkpoint found for warm start.")

        run_uncoated_leave_one_out_cv(
            dataset_dir=dataset_dir,
            checkpoints_dir=checkpoints_dir,
            device=device,
            n_classes=active_n_classes,
            class_names=class_names,
            model_suffix=suffix,
            patch_size=args.uncoated_patch_size,
            patches_per_image=args.uncoated_patches_per_image,
            batch_size=args.uncoated_batch_size,
            num_epochs=args.uncoated_epochs,
            early_stopping_patience=args.uncoated_patience,
            learning_rate=args.uncoated_learning_rate,
            weight_decay=args.uncoated_weight_decay,
            random_seed=args.seed,
            num_workers=0,  # Set to 0 for Windows compatibility
            init_checkpoint_path=init_checkpoint_path,
        )
        return
    
    # Create data loaders - FIXED VERSION
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=dataset_dir,
        batch_size=4,  # Increased from 1 - you have 8GB VRAM
        img_size=512,  # Reduced from 1024 for better batch size
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create model
    model = SimpleUNet(n_classes=active_n_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=active_n_classes,
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Train model
    trainer.train(
        num_epochs=300,  # Can train longer with early stopping
        save_dir=checkpoints_dir,
        early_stopping_patience=50,
        model_suffix=suffix,
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(checkpoints_dir, curves_name)
    )
    
    # Evaluate on test set
    trainer.evaluate_model(test_loader, class_names=class_names)
    
    # Visualize predictions
    visualize_predictions(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        num_samples=4,
        class_names=class_names,
        checkpoint_path=os.path.join(checkpoints_dir, best_model_name)
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best model saved to: {os.path.join(checkpoints_dir, best_model_name)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
