from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

# Support running as a script or within the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from soil_segment.custom_unet import SimpleUNet
    from soil_segment.config import get_data_paths
else:
    from .custom_unet import SimpleUNet
    from .config import get_data_paths


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass(frozen=True)
class ClassStat:
    idx: int
    name: str
    pixel_count: int
    percentage: float


@dataclass(frozen=True)
class NutrientBreakdown:
    totals: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    unmapped_classes: List[str]


# ============================================================================
# CONSTANTS - UPDATED TO MATCH TRAINING SCRIPT
# ============================================================================

# Updated class names to match the training script exactly
DEFAULT_CLASS_NAMES = (
    "background",
    "Black_DAP",
    "Red_MOP",
    "White_AMP",
    "White_Boron",
    "White_Mg",
    "Yellow_Urea_coated",
    "Yellow_Urea_uncoated",
)

# ImageNet normalization (same as training)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

NUTRIENT_KEYS: Tuple[str, ...] = ("N", "P", "K", "S", "Mg", "Br", "Ca")

# Updated nutrient composition database
RAW_MATERIAL_NUTRIENTS: Dict[str, Dict[str, float]] = {
    "urea": {"N": 46.0, "P": 0.0, "K": 0.0, "S": 0.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
    "dap": {"N": 18.0, "P": 46.0, "K": 0.0, "S": 0.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
    "mop": {"N": 0.0, "P": 0.0, "K": 60.0, "S": 0.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
    "amp": {"N": 21.0, "P": 0.0, "K": 0.0, "S": 24.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
    "mg": {"N": 0.0, "P": 0.0, "K": 0.0, "S": 14.0, "Mg": 10.0, "Br": 0.0, "Ca": 0.0},
    "boron": {"N": 0.0, "P": 0.0, "K": 0.0, "S": 0.0, "Mg": 0.0, "Br": 14.0, "Ca": 16.0},
}

# Updated class name mapping to match new naming convention
CLASS_NAME_TO_RAW_MATERIAL: Dict[str, str] = {
    # Yellow Urea variants
    "yellowurea": "urea",
    "yellowureacoated": "urea",
    "yellowureauncoated": "urea",
    "urea": "urea",
    
    # Black DAP variants
    "blackdap": "dap",
    "dap": "dap",
    
    # Red MOP variants
    "redmop": "mop",
    "mop": "mop",
    
    # White AMP variants
    "whiteamp": "amp",
    "amp": "amp",
    "whiteammoniumsulphate": "amp",
    "whiteammoniumsulfate": "amp",
    
    # White Boron variants
    "whiteboron": "boron",
    "boron": "boron",
    
    # White Mg variants
    "whitemg": "mg",
    "mg": "mg",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _normalize_class_name(name: str) -> str:
    """Remove all non-alphanumeric characters and lowercase"""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def calculate_nutrient_breakdown(stats: Sequence[ClassStat]) -> NutrientBreakdown:
    """
    Calculate NPK breakdown from pixel statistics
    
    Updated to use new class naming convention
    """
    # Filter out background (class 0) and zero-count classes
    pellet_stats = [stat for stat in stats if stat.idx != 0 and stat.pixel_count > 0]
    pellet_pixels = sum(stat.pixel_count for stat in pellet_stats)

    per_class: Dict[str, Dict[str, float]] = {}
    totals: Dict[str, float] = {key: 0.0 for key in NUTRIENT_KEYS}
    unmapped: List[str] = []

    if pellet_pixels == 0:
        return NutrientBreakdown(totals=totals, per_class=per_class, unmapped_classes=unmapped)

    for stat in pellet_stats:
        normalized_name = _normalize_class_name(stat.name)
        material_key = CLASS_NAME_TO_RAW_MATERIAL.get(normalized_name)
        
        # Fallback: check if any material key is substring of class name
        if material_key is None:
            for candidate in RAW_MATERIAL_NUTRIENTS:
                if candidate in normalized_name:
                    material_key = candidate
                    break
        
        if material_key is None:
            unmapped.append(stat.name)
            continue

        composition = RAW_MATERIAL_NUTRIENTS[material_key]
        fraction = stat.pixel_count / pellet_pixels
        contributions: Dict[str, float] = {}

        for nutrient in NUTRIENT_KEYS:
            contribution = fraction * composition.get(nutrient, 0.0)
            contributions[nutrient] = contribution
            totals[nutrient] += contribution

        per_class[stat.name] = contributions

    return NutrientBreakdown(totals=totals, per_class=per_class, unmapped_classes=unmapped)


# ============================================================================
# MODEL LOADING AND INFERENCE
# ============================================================================

def resolve_checkpoint_path(path: Optional[Path]) -> Path:
    """
    Resolve checkpoint path with multiple fallback locations
    """
    checkpoints_dir = get_data_paths()["checkpoints"]
    candidates: List[Path] = []

    if path is None:
        candidates.append(checkpoints_dir / "best_model.pth")
    else:
        path = Path(path)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(checkpoints_dir / path)
            candidates.append(PROJECT_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not locate checkpoint. Tried: {tried}")


def _infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer number of classes from checkpoint"""
    for key, value in state_dict.items():
        if key.endswith("final_conv.weight"):
            return int(value.shape[0])
    raise KeyError("Checkpoint missing final_conv.weight; cannot infer classes.")


def load_unet_model(checkpoint_path: Path, device: str) -> Tuple[SimpleUNet, int]:
    """
    Load UNet model from checkpoint
    
    Updated to handle the corrected training checkpoints
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    
    # Handle both direct state_dict and wrapped checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes = _infer_num_classes(state_dict)

    model = SimpleUNet(in_channels=3, n_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    print(f"✓ Loaded UNet with {num_classes} classes")
    
    # Print epoch info if available
    if "epoch" in checkpoint:
        print(f"✓ Checkpoint from epoch {checkpoint['epoch'] + 1}")
    if "best_dice" in checkpoint:
        print(f"✓ Best validation Dice: {checkpoint['best_dice']:.4f}")

    return model, num_classes


def build_unet_transform(img_size: int) -> transforms.Compose:
    """
    Build transform matching the training script exactly
    
    CRITICAL: Must match the JointTransform used in training (without augmentation)
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def load_img_as_rgb(img_path: str) -> np.ndarray:
    """Load image and convert BGR to RGB"""
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_with_unet(
    image_path: str,
    model: SimpleUNet,
    transform: transforms.Compose,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UNet inference on a single image
    
    Returns:
        pred_mask: Predicted segmentation mask (H, W) with class indices
        rgb_image: Original RGB image
    """
    rgb_image = load_img_as_rgb(image_path)
    pil_image = Image.fromarray(rgb_image.astype(np.uint8))
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Handle both binary and multi-class segmentation
    if output.shape[1] == 1:
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    else:
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Resize back to original dimensions
    original_size = rgb_image.shape[:2]
    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return pred_mask_resized, rgb_image


# ============================================================================
# CLASS NAME LOADING
# ============================================================================

def load_class_names(num_classes: int, class_json: Optional[Path] = None) -> List[str]:
    """
    Load class names with proper fallback logic
    
    Updated to match training script defaults
    """
    class_file = class_json or (PROJECT_ROOT / "datasets" / "annotate" / "classes.json")
    names: List[str] = []

    if class_file.exists():
        try:
            data = json.loads(class_file.read_text())
            names = data.get("class_names", []) or []
            print(f"✓ Loaded {len(names)} class names from {class_file}")
        except json.JSONDecodeError:
            print(f"⚠ Failed to parse {class_file}, using defaults")
            names = []

    if not names:
        names = list(DEFAULT_CLASS_NAMES)
        print(f"✓ Using default class names ({len(names)} classes)")

    # Ensure we have enough names
    if len(names) < num_classes:
        names.extend(f"Class_{idx}" for idx in range(len(names), num_classes))
        print(f"⚠ Extended class names to {num_classes} classes")

    return names[:num_classes]


# ============================================================================
# NPK CALCULATION
# ============================================================================

def build_class_stats(mask: np.ndarray, class_names: Sequence[str]) -> List[ClassStat]:
    """Build statistics for each class in the mask"""
    total_pixels = mask.size
    stats: List[ClassStat] = []
    
    for idx, name in enumerate(class_names):
        pixel_count = int(np.sum(mask == idx))
        percentage = (pixel_count / total_pixels) * 100 if total_pixels else 0.0
        stats.append(
            ClassStat(
                idx=idx,
                name=name,
                pixel_count=pixel_count,
                percentage=percentage
            )
        )
    
    return stats


def get_npk_from_mask(
    mask: np.ndarray,
    class_names: Sequence[str],
    include_all_nutrients: bool = False
) -> List[float]:
    """
    Calculate NPK (or N,P,K,S,Mg,Br,Ca) from segmentation mask
    
    Args:
        mask: Segmentation mask with class indices
        class_names: List of class names matching indices
        include_all_nutrients: If True, return all 7 nutrients; else just N,P,K
    
    Returns:
        List of nutrient percentages
    """
    stats = build_class_stats(mask, class_names)
    breakdown = calculate_nutrient_breakdown(stats)
    
    if breakdown.unmapped_classes:
        print(f"⚠ Warning: Unmapped classes: {breakdown.unmapped_classes}")
    
    if include_all_nutrients:
        return [float(breakdown.totals[key]) for key in NUTRIENT_KEYS]
    else:
        return [float(breakdown.totals[key]) for key in ("N", "P", "K")]


# ============================================================================
# DATASET HANDLING
# ============================================================================

def _parse_npk_from_path(path_str: str) -> List[int]:
    """
    Parse NPK values from directory name
    
    Expected format: .../N-P-K/image.jpg
    Example: .../10-15-20/sample1.jpg → [10, 15, 20]
    """
    p = Path(path_str)
    
    # Check parent and grandparent directories
    for candidate in (p.parent, p.parent.parent):
        parts = candidate.name.split("-")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return list(map(int, parts))
    
    raise ValueError(f"Could not parse NPK from path: {path_str}")


def _collect_image_paths(dataset_dir: Path) -> List[List[str]]:
    """
    Collect image paths organized by NPK formula
    
    Structure expected:
        dataset_dir/
            10-15-20/
                image1.jpg
                image2.jpg
            15-20-10/
                image1.jpg
    
    Returns:
        List of lists, where each inner list contains paths for one NPK formula
    """
    img_path_list: List[List[str]] = []
    subdirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]

    for subdir in subdirs:
        # Try direct image files
        image_files = [
            p for p in sorted(subdir.iterdir()) 
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        # Try images/ subdirectory
        if not image_files:
            images_dir = subdir / "images"
            if images_dir.is_dir():
                image_files = [
                    p for p in sorted(images_dir.iterdir())
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                ]

        if image_files:
            img_path_list.append([str(p) for p in image_files])

    return img_path_list


def auto_detect_image_folders(base_path: Optional[Path]) -> Tuple[List[List[str]], Optional[Path]]:
    """
    Auto-detect regression dataset with multiple fallback paths
    
    Updated to be more robust and informative
    """
    img_path_list: List[List[str]] = []
    candidate_paths: List[Path] = []
    seen: set[Path] = set()

    def _register_candidate(path_candidate: Path) -> None:
        path_candidate = path_candidate.expanduser()
        if path_candidate in seen:
            return
        seen.add(path_candidate)
        candidate_paths.append(path_candidate)

    if base_path is None:
        # Use config paths
        data_paths = get_data_paths()
        _register_candidate(data_paths["regression_dataset"])
        _register_candidate(PROJECT_ROOT / "datasets" / "regression_dataset")
        _register_candidate(PROJECT_ROOT / "datasets" / "regressor_dataset")
    else:
        base_path = Path(base_path)
        _register_candidate(base_path)
        
        if not base_path.is_absolute():
            _register_candidate(Path.cwd() / base_path)
            _register_candidate(PROJECT_ROOT / base_path)
            _register_candidate(PROJECT_ROOT / "datasets" / base_path)
        
        # Add variant names
        if base_path.name == "regression_dataset":
            _register_candidate(PROJECT_ROOT / "datasets" / "regressor_dataset")

    resolved_base: Optional[Path] = None
    
    print("\nSearching for regression dataset...")
    for candidate in candidate_paths:
        print(f"  Checking: {candidate}")
        if candidate.exists() and candidate.is_dir():
            img_path_list = _collect_image_paths(candidate)
            if img_path_list:
                resolved_base = candidate.resolve()
                print(f"  ✓ Found {len(img_path_list)} NPK formulas")
                break
        else:
            print(f"    (not found)")

    if not resolved_base:
        print("  ✗ No valid dataset found")

    return img_path_list, resolved_base


# ============================================================================
# REGRESSION DATA PREPARATION
# ============================================================================

def prepare_regression_data(
    img_path_list: List[List[str]],
    model: SimpleUNet,
    transform: transforms.Compose,
    device: str,
    class_names: Sequence[str],
    *,
    max_images: Optional[int] = None,
    skip_errors: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Process all images and prepare regression training data
    
    Args:
        img_path_list: Nested list of image paths (grouped by NPK formula)
        model: Loaded UNet model
        transform: Image transform
        device: Computation device
        class_names: List of class names
        max_images: Limit total images processed (for testing)
        skip_errors: Continue on errors vs raise
        verbose: Print detailed progress
    
    Returns:
        approx_npk: Predicted NPK values from segmentation [N, 3]
        actual_npk: Ground truth NPK from directory names [N, 3]
        used_paths: List of successfully processed image paths
    """
    approx_npk: List[List[float]] = []
    actual_npk: List[List[int]] = []
    used_paths: List[str] = []

    # Flatten nested list
    flattened = [path for paths in img_path_list for path in paths]
    
    if max_images is not None:
        flattened = flattened[:max_images]
        if verbose:
            print(f"Limiting to {max_images} images for testing")

    print(f"\n{'='*70}")
    print(f"Processing {len(flattened)} images for regression data")
    print(f"{'='*70}\n")

    errors = []
    
    for path in tqdm(flattened, desc="Computing NPK features", disable=not verbose):
        try:
            # Run segmentation
            pred_mask, _ = predict_with_unet(path, model, transform, device)
            
            # Calculate nutrients from mask
            npk_approx = get_npk_from_mask(pred_mask, class_names, include_all_nutrients=False)
            
            # Parse ground truth from path
            npk_actual = _parse_npk_from_path(path)
            
            approx_npk.append(npk_approx)
            actual_npk.append(npk_actual)
            used_paths.append(path)
            
        except Exception as exc:
            error_msg = f"Skipping {Path(path).name}: {exc}"
            errors.append(error_msg)
            
            if not skip_errors:
                raise

    # Report errors
    if errors:
        print(f"\n⚠ Encountered {len(errors)} errors:")
        for err in errors[:5]:  # Show first 5
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not approx_npk:
        raise RuntimeError(
            "No regression samples prepared. Check dataset paths and labels."
        )

    print(f"\n✓ Successfully processed {len(approx_npk)}/{len(flattened)} images")

    return (
        np.asarray(approx_npk, dtype=np.float32),
        np.asarray(actual_npk, dtype=np.float32),
        used_paths,
    )


# ============================================================================
# REGRESSION MODEL
# ============================================================================

def build_npk_regressor(
    degree: int = 1,
    use_ridge: bool = True,
    alpha: float = 1.0,
    scale_features: bool = True
):
    """
    Build regression pipeline
    
    Args:
        degree: Polynomial degree (1=linear, 2=quadratic, etc.)
        use_ridge: Use Ridge regression (L2 regularization) vs plain LinearRegression
        alpha: Regularization strength (only used if use_ridge=True)
        scale_features: Standardize features before regression
    
    Returns:
        sklearn Pipeline
    """
    steps = []
    
    # Optional feature scaling
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    # Polynomial features
    if degree > 1:
        steps.append(('poly', PolynomialFeatures(degree=degree, include_bias=False)))
    
    # Regression model
    if use_ridge:
        steps.append(('regressor', Ridge(alpha=alpha)))
    else:
        steps.append(('regressor', LinearRegression()))
    
    if not steps:
        raise ValueError("No regression steps configured.")

    return Pipeline(steps) if len(steps) > 1 else steps[0][1]


def train_regressor_with_holdout(
    y_approx: Sequence[Sequence[float]],
    y_true: Sequence[Sequence[float]],
    *,
    save_path: Optional[Path] = None,
    degree: int = 1,
    use_ridge: bool = True,
    alpha: float = 1.0,
    scale_features: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train regression model with validation split
    
    Returns:
        regressor: Trained model
        metrics: Dict of validation metrics
        holdout_results: Tuple of (predictions, actuals) on validation set
    """
    # Prepare arrays
    features = np.asarray(y_approx, dtype=np.float32)
    targets = np.asarray(y_true, dtype=np.float32)
    
    if features.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: approx {features.shape} vs actual {targets.shape}"
        )
    
    if len(features) < 2:
        raise ValueError("Need at least 2 samples for validation split")

    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    print(f"\n{'='*70}")
    print(f"Regression Training Configuration")
    print(f"{'='*70}")
    print(f"Total samples: {len(features)}")
    print(f"Train samples: {len(x_train)}")
    print(f"Val samples: {len(x_val)}")
    print(f"Polynomial degree: {degree}")
    print(f"Regularization: {'Ridge' if use_ridge else 'None'} (alpha={alpha})")
    print(f"Feature scaling: {scale_features}")
    print(f"{'='*70}\n")

    # Train on validation split
    validation_model = build_npk_regressor(
        degree=degree,
        use_ridge=use_ridge,
        alpha=alpha,
        scale_features=scale_features
    )
    validation_model.fit(x_train, y_train)
    val_predictions = validation_model.predict(x_val)

    # Calculate validation metrics
    val_mae = mean_absolute_error(y_val, val_predictions, multioutput='raw_values')
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions, multioutput='raw_values'))
    val_r2 = r2_score(y_val, val_predictions, multioutput='raw_values')

    # Train final model on all data
    final_model = build_npk_regressor(
        degree=degree,
        use_ridge=use_ridge,
        alpha=alpha,
        scale_features=scale_features
    )
    final_model.fit(features, targets)

    # Save model
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, save_path)
        print(f"✓ Model saved to: {save_path}")

    metrics = {
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "val_sample_count": len(x_val),
        "train_sample_count": len(x_train),
    }
    
    return final_model, metrics, (val_predictions, y_val)


def print_regression_metrics(
    metrics: Dict,
    nutrient_labels: List[str] = ["N", "P", "K"]
):
    """Print formatted regression metrics"""
    print(f"\n{'='*70}")
    print(f"Validation Metrics")
    print(f"{'='*70}")
    print(f"Train samples: {metrics['train_sample_count']}")
    print(f"Val samples: {metrics['val_sample_count']}")
    print(f"{'-'*70}")
    
    print(f"{'Nutrient':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print(f"{'-'*70}")
    
    for label, mae, rmse, r2 in zip(
        nutrient_labels,
        metrics["val_mae"],
        metrics["val_rmse"],
        metrics["val_r2"]
    ):
        print(f"{label:<10} {mae:<12.4f} {rmse:<12.4f} {r2:<12.4f}")
    
    print(f"{'='*70}\n")


def print_sample_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    nutrient_labels: List[str] = ["N", "P", "K"],
    max_samples: int = 10
):
    """Print sample predictions vs actuals"""
    print(f"\n{'='*70}")
    print(f"Sample Predictions (showing up to {max_samples})")
    print(f"{'='*70}")
    
    n_samples = min(max_samples, len(actuals))
    
    for i in range(n_samples):
        pred_str = " ".join(f"{p:6.2f}" for p in predictions[i])
        actual_str = " ".join(f"{a:6.2f}" for a in actuals[i])
        error_str = " ".join(f"{abs(p-a):6.2f}" for p, a in zip(predictions[i], actuals[i]))
        
        print(f"Sample {i+1:3d}:")
        print(f"  Predicted: [{pred_str}]")
        print(f"  Actual:    [{actual_str}]")
        print(f"  Error:     [{error_str}]")
    
    if len(actuals) > n_samples:
        print(f"\n... and {len(actuals) - n_samples} more samples")
    
    print(f"{'='*70}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NPK regression model from UNet segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to regression dataset directory",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to UNet checkpoint (.pth). Defaults to checkpoints/best_model.pth",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save regression_model.pkl",
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Image size for UNet inference (should match training)",
    )
    
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Polynomial degree (1=linear, 2=quadratic)",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength",
    )
    
    parser.add_argument(
        "--no-ridge",
        action="store_true",
        help="Use LinearRegression instead of Ridge",
    )
    
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        help="Disable feature standardization",
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split size (0-1)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images (for testing)",
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first error",
    )
    
    return parser.parse_args(argv)


# ============================================================================
# MAIN
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main regression training pipeline"""
    args = parse_args(argv)
    
    # Setup device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"NPK Regression Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    # Find dataset
    img_path_list, dataset_root = auto_detect_image_folders(args.dataset)
    if not img_path_list or dataset_root is None:
        print("✗ No valid image folders found for regression training")
        return 1

    print(f"✓ Using dataset: {dataset_root}")
    print(f"✓ Found {len(img_path_list)} NPK formulas")
    total_images = sum(len(paths) for paths in img_path_list)
    print(f"✓ Total images: {total_images}")

    # Load UNet model
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, num_classes = load_unet_model(checkpoint_path, device)

    # Load class names
    class_names = load_class_names(num_classes)
    print(f"✓ Using {len(class_names)} classes: {class_names}")

    # Build transform
    transform = build_unet_transform(args.img_size)
    print(f"✓ Image size for inference: {args.img_size}x{args.img_size}")

    # Prepare regression data
    approx_npk, actual_npk, used_paths = prepare_regression_data(
        img_path_list,
        model,
        transform,
        device,
        class_names,
        max_images=args.max_images,
        skip_errors=not args.strict,
        verbose=True,
    )

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = get_data_paths()["checkpoints"] / "regression_model.pkl"

    # Train regressor
    regressor, metrics, holdout_results = train_regressor_with_holdout(
        approx_npk,
        actual_npk,
        save_path=output_path,
        degree=args.degree,
        use_ridge=not args.no_ridge,
        alpha=args.alpha,
        scale_features=not args.no_scaling,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Print results
    print_regression_metrics(metrics)
    
    val_predictions, val_actuals = holdout_results
    print_sample_predictions(val_predictions, val_actuals)

    # Test on all data
    all_predictions = regressor.predict(approx_npk)
    
    print("\n" + "="*70)
    print("Full Dataset Performance (Train + Val)")
    print("="*70)
    
    full_mae = mean_absolute_error(actual_npk, all_predictions, multioutput='raw_values')
    full_rmse = np.sqrt(mean_squared_error(actual_npk, all_predictions, multioutput='raw_values'))
    full_r2 = r2_score(actual_npk, all_predictions, multioutput='raw_values')
    
    for label, mae, rmse, r2 in zip(["N", "P", "K"], full_mae, full_rmse, full_r2):
        print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    print("="*70 + "\n")

    print("✓ Training complete!")
    print(f"✓ Model saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
