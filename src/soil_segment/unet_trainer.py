import sys
from pathlib import Path

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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import random

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

        # Find corresponding mask
        mask_path = os.path.join(self.mask_dir, img_name)
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    break
            else:
                raise FileNotFoundError(
                    f"No mask found for image {img_name} in {self.mask_dir}"
                )

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
        # Get original data without transform
        original_idx = self.indices[idx]
        img_name = self.dataset.images[original_idx]
        img_path = os.path.join(self.dataset.image_dir, img_name)
        
        # Find mask
        mask_path = os.path.join(self.dataset.mask_dir, img_name)
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                mask_path = os.path.join(self.dataset.mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    break
        
        # Load fresh
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Convert mask
        mask_array = np.array(mask)
        if len(mask_array.shape) > 2:
            mask_array = mask_array[:, :, 0]
        mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')
        
        # Apply this subset's specific transform
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask


# ============================================================================
# METRICS
# ============================================================================
class DiceScore(nn.Module):
    """Dice coefficient for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
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
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        return torch.stack(dice_scores).mean()


class IoUScore(nn.Module):
    """Intersection over Union for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        iou_scores = []
        for i in range(self.num_classes):
            pred_i = (pred == i)
            target_i = (target == i)
            
            intersection = (pred_i & target_i).float().sum()
            union = (pred_i | target_i).float().sum()
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou)
        
        return torch.stack(iou_scores).mean()


# ============================================================================
# TRAINER CLASS
# ============================================================================
class SegmentationTrainer:
    """Training pipeline for UNet with proper handling of small datasets"""
    
    def __init__(self, model, device, train_loader, val_loader, n_classes=7, 
                 learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.dice_metric = DiceScore(n_classes)
        self.iou_metric = IoUScore(n_classes)
        
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
    
    def train(self, num_epochs, save_dir='checkpoints', early_stopping_patience=50):
        """Full training loop with early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        
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
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_dice': self.best_dice,
                    'history': self.history
                }, os.path.join(save_dir, 'best_model.pth'))
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
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Print progress
            epoch_time = time.time() - epoch_start
            status = f"Epoch {epoch+1:3d}/{num_epochs}"
            status += f" | Train L:{train_loss:.4f} D:{train_dice:.4f} IoU:{train_iou:.4f}"
            status += f" | Val L:{val_loss:.4f} D:{val_dice:.4f} IoU:{val_iou:.4f}"
            status += f" | LR:{current_lr:.6f} | {epoch_time:.1f}s"
            
            if is_best:
                status += " ⭐ BEST"
            
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
            class_names = [f'Class {i}' for i in range(self.n_classes)]
        
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
        print(classification_report(
            all_targets, 
            all_predictions, 
            target_names=class_names,
            zero_division=0
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
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
    n_classes = 7
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    if class_names is None:
        class_names = [
            'Black_DAP', 'Red_MOP', 'White_AMP', 'White_Boron',
            'White_Mg', 'Yellow_Urea_coated', 'Yellow_Urea_uncoated'
        ]
    
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


def verify_dataset_classes(data_dir, expected_classes=7):
    """Verify that all masks have the correct class labels"""
    print("\n" + "="*70)
    print("Verifying Dataset Classes")
    print("="*70)
    
    mask_dir = os.path.join(data_dir, 'masks')
    mask_files = sorted([f for f in os.listdir(mask_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
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
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"\nFound classes: {sorted(all_classes)}")
    print(f"Expected classes: {list(range(expected_classes))}")
    
    if set(all_classes) == set(range(expected_classes)):
        print("✓ All classes present and correct!")
    else:
        missing = set(range(expected_classes)) - all_classes
        extra = all_classes - set(range(expected_classes))
        if missing:
            print(f"⚠ Missing classes: {missing}")
        if extra:
            print(f"⚠ Unexpected classes: {extra}")
    
    print("\nClass distribution across dataset:")
    for cls in sorted(class_counts.keys()):
        print(f"  Class {cls}: appears in {class_counts[cls]} masks")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
def main():
    """Main training pipeline"""
    
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
    dataset_dir = str(paths["unet_dataset"])
    checkpoints_dir = str(paths["checkpoints"])
    
    # Verify dataset first
    verify_dataset_classes(dataset_dir, expected_classes=7)
    
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
    model = SimpleUNet(n_classes=7)
    
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
        n_classes=7,
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Train model
    trainer.train(
        num_epochs=300,  # Can train longer with early stopping
        save_dir=checkpoints_dir,
        early_stopping_patience=50
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(checkpoints_dir, 'training_curves.png')
    )
    
    # Evaluate on test set
    class_names = [
        'Black_DAP', 'Red_MOP', 'White_AMP', 'White_Boron',
        'White_Mg', 'Yellow_Urea_coated', 'Yellow_Urea_uncoated'
    ]
    trainer.evaluate_model(test_loader, class_names=class_names)
    
    # Visualize predictions
    visualize_predictions(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        num_samples=4,
        class_names=class_names,
        checkpoint_path=os.path.join(checkpoints_dir, 'best_model.pth')
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best model saved to: {os.path.join(checkpoints_dir, 'best_model.pth')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()