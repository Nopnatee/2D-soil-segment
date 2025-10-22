import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------------
# Embedded U-Net Architecture
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=8, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)  # logits


# -----------------------------
# Data pipeline and training
# -----------------------------
class AlbumentationsDataset(Dataset):
    """Dataset with Albumentations transforms"""

    def __init__(self, image_dir, mask_dir, transform=None, debug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.debug = debug
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if self.debug:
            print(f"[Dataset] Found {len(self.images)} images in {image_dir}")
            print(f"[Dataset] First few images: {self.images[:3]}")

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
                raise FileNotFoundError(f"No mask found for image {img_name}")

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        # Convert multi-channel mask to single channel if needed
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if self.debug and idx < 3:
            print(f"\n[Dataset DEBUG] Sample {idx}:")
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"Mask unique values: {np.unique(mask)}")

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        if self.debug and idx < 3:
            print(f"[Dataset DEBUG] After transform:")
            print(f"Image tensor shape: {image.shape}, dtype: {image.dtype}")
            print(f"Mask tensor shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"Mask unique values: {torch.unique(mask)}")

        return image, mask.long()


def get_training_augmentation(img_size=1024):
    """
    Training augmentation pipeline with Albumentations
    Matches the structure from your image
    """
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=img_size * 2, p=1.0),
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Transpose(p=1.0),
            A.Rotate(limit=90, p=1.0),
            A.Rotate(limit=180, p=1.0),
            A.Rotate(limit=270, p=1.0),
            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=90, p=1.0)], p=1.0),
            A.Compose([A.VerticalFlip(p=1.0), A.Rotate(limit=90, p=1.0)], p=1.0),
        ], p=1.0),
        A.Rotate(limit=30, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.3
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(),
    ])

    return train_transform


def get_validation_augmentation(img_size=512):
    """
    Validation augmentation pipeline (no augmentation, just preprocessing)
    """
    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(),
    ])

    return val_transform


class DiceScore(nn.Module):
    """Dice coefficient for segmentation evaluation"""

    def __init__(self, num_classes, smooth=1e-6, debug=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.debug = debug
        self.call_count = 0

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        dice_scores = []

        if self.debug and self.call_count < 5:
            print(f"\n=== DICE DEBUG CALL {self.call_count} ===")
            print(f"Pred shape: {pred.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Target unique values: {torch.unique(target)}")

            pred_classes = torch.argmax(pred, dim=1)
            print(f"Predicted classes unique: {torch.unique(pred_classes)}")

        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = (target == i).float()

            dice = (
                2 * (pred_i * target_i).sum() + self.smooth
            ) / (pred_i.sum() + target_i.sum() + self.smooth)
            dice_scores.append(dice)

            if self.debug and self.call_count < 5:
                num = (2 * (pred_i * target_i).sum() + self.smooth).item()
                den = (pred_i.sum() + target_i.sum() + self.smooth).item()
                print(f"Class {i}: numerator={num:.4f}, denominator={den:.4f}, dice={dice.item():.4f}")

        if self.debug and self.call_count < 5:
            print("=" * 30)

        self.call_count += 1
        return torch.stack(dice_scores).mean()


class SegmentationTrainer:
    """Complete training pipeline for U-Net with Albumentations"""

    def __init__(self, model, device, train_loader, val_loader, n_classes=4, debug=False):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_classes = n_classes
        self.debug = debug

        # Multi-GPU support
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = nn.DataParallel(self.model)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.dice_metric = DiceScore(n_classes, debug=debug)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'lr': []
        }

        # Best model tracking
        self.best_dice = 0.0
        self.best_model_state = None
        self.previous_lr = self.optimizer.param_groups[0]['lr']
        # Epoch tracking and improvement log
        self.current_epoch = 0
        self.best_epoch = None
        # List of dicts: each time we get a new best, store the epoch and score
        self.best_dice_progress = []

    def debug_batch(self, data, target, output, batch_idx, phase="train"):
        """Debug a single batch"""
        if batch_idx == 0:
            print(f"\n=== {phase.upper()} BATCH DEBUG ===")
            print(f"Input shape: {data.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Target unique values: {torch.unique(target)}")

            pred_classes = torch.argmax(output, dim=1)
            print(f"Predicted classes unique: {torch.unique(pred_classes)}")

            pred_probs = F.softmax(output, dim=1)
            for i in range(self.n_classes):
                class_prob = pred_probs[:, i].mean()
                print(f"Class {i} avg probability: {class_prob:.4f}")

            print("=" * 40)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Debug first batch
            if self.debug and batch_idx == 0:
                self.debug_batch(data, target, output, batch_idx, "train")

            # Calculate loss
            loss = self.criterion(output, target)
            dice = self.dice_metric(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            running_loss += loss.item()
            running_dice += dice.item()

        avg_loss = running_loss / len(self.train_loader)
        avg_dice = running_dice / len(self.train_loader)

        return avg_loss, avg_dice

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                output = self.model(data)

                # Debug first batch
                if self.debug and batch_idx == 0:
                    self.debug_batch(data, target, output, batch_idx, "val")

                loss = self.criterion(output, target)
                dice = self.dice_metric(output, target)

                running_loss += loss.item()
                running_dice += dice.item()

        avg_loss = running_loss / len(self.val_loader)
        avg_dice = running_dice / len(self.val_loader)

        return avg_loss, avg_dice

    def train(self, num_epochs, save_dir='checkpoints', print_every=1):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)

        print(f"Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Classes: {self.n_classes}")
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.current_epoch = epoch + 1

            # Enable debug only for first 5 epochs
            self.debug = (epoch < 5)
            self.dice_metric.debug = (epoch < 5)

            # Training phase
            train_loss, train_dice = self.train_epoch()

            # Validation phase
            val_loss, val_dice = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Check for LR reduction
            lr_reduced = current_lr != self.previous_lr
            if lr_reduced:
                self.previous_lr = current_lr

            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['lr'].append(current_lr)

            # Save best model
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.best_epoch = self.current_epoch
                self.best_model_state = self.model.state_dict().copy()
                # Log the improvement
                self.best_dice_progress.append({
                    'epoch': self.current_epoch,
                    'val_dice': float(self.best_dice)
                })
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'best_epoch': self.best_epoch,
                    'best_progress': self.best_dice_progress,
                    'history': self.history
                }, os.path.join(save_dir, 'best_model.pth'))

            # Regular checkpoint
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            # Print summary
            if (epoch + 1) % print_every == 0:
                epoch_time = time.time() - epoch_start
                status = ""
                if is_best:
                    status += " [BEST]"
                if lr_reduced:
                    status += f" [LR: {current_lr:.6f}]"

                print(f"Epoch {epoch+1:4d}/{num_epochs} | "
                      f"Train L: {train_loss:.4f} D: {train_dice:.4f} | "
                      f"Val L: {val_loss:.4f} D: {val_dice:.4f} | "
                      f"Time: {epoch_time:.1f}s{status}")
                if is_best:
                    print(f"  ↳ New best Dice {val_dice:.4f} at epoch {self.current_epoch}")

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation Dice: {self.best_dice:.4f}")
        if self.best_epoch is not None:
            print(f"Best at epoch: {self.best_epoch}")
            # Show progression of improvements (epoch -> dice)
            print("Dice improvements by epoch:")
            for entry in self.best_dice_progress:
                print(f"  epoch {entry['epoch']:4d} -> {entry['val_dice']:.4f}")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Best model loaded.")

    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Dice curves
        axes[0, 1].plot(self.history['train_dice'], label='Train Dice')
        axes[0, 1].plot(self.history['val_dice'], label='Val Dice')
        axes[0, 1].set_title('Dice Score Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Combined metrics
        axes[1, 1].plot(self.history['val_loss'], label='Val Loss', alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.history['val_dice'], label='Val Dice', color='orange', alpha=0.7)
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='blue')
        ax2.set_ylabel('Dice Score', color='orange')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()

    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                output = self.model(data)
                pred = torch.argmax(output, dim=1)

                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions))

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.show()

        return all_predictions, all_targets


def create_data_loaders(data_dir, batch_size=4, img_size=1024,
                        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                        random_seed=42, num_workers=4, debug=False):
    """
    Create data loaders with Albumentations transforms
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Get training and validation transforms
    train_transform = get_training_augmentation(img_size)
    val_transform = get_validation_augmentation(img_size)

    # Create full dataset to get indices
    temp_dataset = AlbumentationsDataset(
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'masks'),
        transform=None,
        debug=False
    )

    total_samples = len(temp_dataset)
    indices = list(range(total_samples))

    # Split indices
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_indices, temp_indices = train_test_split(
        indices, train_size=train_size,
        random_state=random_seed, shuffle=True
    )
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_size,
        random_state=random_seed, shuffle=True
    )

    # Create datasets with appropriate transforms
    train_dataset = AlbumentationsDataset(
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'masks'),
        transform=train_transform,
        debug=debug
    )

    val_dataset = AlbumentationsDataset(
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'masks'),
        transform=val_transform,
        debug=False
    )

    test_dataset = AlbumentationsDataset(
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'masks'),
        transform=val_transform,
        debug=False
    )

    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # Create data loaders
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Train: {len(train_indices)} ({train_ratio*100:.1f}%)")
    print(f"  Val: {len(val_indices)} ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_indices)} ({test_ratio*100:.1f}%)")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Batch size: {batch_size}")
    print()

    return train_loader, val_loader, test_loader


def visualize_augmentations(dataset, num_samples=5):
    """Visualize augmented samples"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))

    for i in range(num_samples):
        image, mask = dataset[i]

        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_vis = image * std + mean
        image_vis = torch.clamp(image_vis, 0, 1)

        # Plot
        axes[i, 0].imshow(image_vis.permute(1, 2, 0))
        axes[i, 0].set_title(f'Augmented Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='tab20')
        axes[i, 1].set_title(f'Ground Truth Mask {i+1}')
        axes[i, 1].axis('off')

        # Show unique classes
        unique_classes = torch.unique(mask)
        axes[i, 2].text(0.5, 0.5, f'Classes: {unique_classes.tolist()}',
                       ha='center', va='center', fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150)
    plt.show()


def visualize_predictions(model, dataset, device, num_samples=5):
    """Visualize model predictions"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))

    for i in range(num_samples):
        image, mask = dataset[i]
        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze().cpu()

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_vis = image * std + mean
        image_vis = torch.clamp(image_vis, 0, 1)

        # Plot
        axes[i, 0].imshow(image_vis.permute(1, 2, 0))
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='tab20')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='tab20')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()


def main():
    """Main training script"""
    # Configuration
    IMG_SIZE = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 200
    NUM_CLASSES = 8
    DATA_DIR = 'UNET_dataset'
    NUM_WORKERS = 4  # Adjust based on your CPU

    # Set device and optimize for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Create model
    model = SimpleUNet(n_classes=NUM_CLASSES)
    print(f"Model created with {NUM_CLASSES} classes\n")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        num_workers=NUM_WORKERS,
        debug=False
    )

    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=NUM_CLASSES,
        debug=False
    )

    # Train model
    print("Starting training...")
    trainer.train(num_epochs=NUM_EPOCHS, print_every=1)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.evaluate_model(test_loader)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

