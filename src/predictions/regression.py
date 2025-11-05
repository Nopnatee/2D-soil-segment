"""
Utilities for calibrating U-Net segmentation outputs to ground-truth NPK labels.

This module rewrites the regression-focused notebook into reusable Python code.
It provides helpers for loading the segmentation model, extracting pellet areas
per image, converting those areas into approximate NPK ratios, and training or
loading a regression model that maps the approximations to true fertilizer
composition values.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

try:
    from ..soil_segment.custom_unet import SimpleUNet
except ImportError:  # executed when module is run as a script
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from soil_segment.custom_unet import SimpleUNet

logger = logging.getLogger(__name__)

NUTRIENT_KEYS: Tuple[str, ...] = ("N", "P", "K")


@dataclass(frozen=True)
class MaskDefinition:
    """Description of a pellet mask that maps Unet class IDs to nutrient content."""

    index: int
    label: str
    composition: Dict[str, float]


# Default mapping matches the current seven-class U-Net (background + six pellet classes).
DEFAULT_MASK_DEFINITIONS: Tuple[MaskDefinition, ...] = (
    MaskDefinition(index=1, label="Black_DAP", composition={"N": 18.0, "P": 45.5, "K": 0.0}),
    MaskDefinition(index=2, label="Red_MOP", composition={"N": 0.0, "P": 0.0, "K": 60.0}),
    MaskDefinition(index=3, label="White_AMP", composition={"N": 20.5, "P": 0.0, "K": 0.0}),
    MaskDefinition(index=4, label="White_Boron", composition={"N": 0.0, "P": 0.0, "K": 0.0}),
    MaskDefinition(index=5, label="White_Mg", composition={"N": 0.0, "P": 0.0, "K": 0.0}),
    MaskDefinition(index=6, label="Yellow_Urea", composition={"N": 46.0, "P": 0.0, "K": 0.0}),
)


@dataclass
class RegressionConfig:
    """Container for regression-training configuration."""

    dataset_dir: Path = Path("datasets/regressor_dataset")
    checkpoint_path: Optional[Path] = None
    regressor_output: Path = Path("checkpoints/regression_model.pkl")
    num_classes: int = 7
    image_size: int = 512
    device: Optional[str] = None
    polynomial_degree: int = 2
    ridge_alpha: float = 1.0
    mask_definitions: Sequence[MaskDefinition] = DEFAULT_MASK_DEFINITIONS
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    show_progress: bool = True
    generate_visualizations: bool = False
    visualization_output: Optional[Path] = None


@dataclass
class RegressionSample:
    """Feature/label extracted from a single image."""

    image_path: Path
    cluster_areas: np.ndarray  # shape (len(mask_definitions),)
    approx_npk: np.ndarray  # shape (3,)
    actual_npk: Optional[np.ndarray]  # shape (3,) or None when unavailable


def resolve_checkpoint_path(filename: str = "best_model.pth", search_start: Optional[Path] = None) -> Path:
    """Locate a checkpoint file by walking up from the provided directory."""

    start_dir = Path(search_start or Path.cwd()).resolve()
    for candidate_root in (start_dir, *start_dir.parents):
        candidate = candidate_root / "checkpoints" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint '{filename}' not found relative to {start_dir}")


def determine_device(requested: Optional[str]) -> torch.device:
    """Choose the torch device, respecting an explicit request when available."""

    if requested:
        return torch.device(requested)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_unet_model(
    checkpoint_path: Optional[Path] = None,
    *,
    num_classes: int = 7,
    device: Optional[torch.device] = None,
) -> SimpleUNet:
    """Load the trained U-Net segmentation model."""

    resolved_device = device or determine_device(None)
    checkpoint = checkpoint_path or resolve_checkpoint_path()
    checkpoint = Path(checkpoint)

    state = torch.load(checkpoint, map_location=resolved_device)
    state_dict = state.get("model_state_dict", state)

    model = SimpleUNet(in_channels=3, n_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    logger.info("Loaded U-Net checkpoint from %s", checkpoint)
    return model


def build_unet_transform(image_size: int) -> transforms.Compose:
    """Create the preprocessing transform expected by the segmentation model."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def auto_detect_image_folders(
    base_path: Path,
    *,
    image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> List[List[Path]]:
    """
    Discover sub-directories that contain images for regression training.

    Returns:
        A list of per-folder image-path lists, mirroring the original notebook behaviour.
    """

    base_path = Path(base_path)
    if not base_path.exists():
        logger.warning("Regression dataset directory does not exist: %s", base_path)
        return []

    folders: List[List[Path]] = []
    for subdir in sorted(p for p in base_path.iterdir() if p.is_dir()):
        images = sorted(
            file for file in subdir.iterdir() if file.suffix.lower() in image_extensions and file.is_file()
        )
        if images:
            folders.append(images)
        else:
            logger.debug("Skipping folder without images: %s", subdir)

    return folders


def load_img_as_rgb(image_path: Path) -> np.ndarray:
    """Load an image from disk and convert it to RGB order."""

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_with_unet(
    image_path: Path,
    model: SimpleUNet,
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the segmentation model on an image path.

    Returns:
        pred_mask: HxW array of class indices.
        rgb_image: HxWx3 numpy array for further analysis.
    """

    rgb_image = load_img_as_rgb(image_path)
    pil_image = Image.fromarray(rgb_image.astype(np.uint8))

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)

    if logits.shape[1] == 1:
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (probabilities > 0.5).astype(np.uint8)
    else:
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    height, width = rgb_image.shape[:2]
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    return pred_mask, rgb_image


def get_cluster_mask_gpu(
    image_path: Path,
    model: SimpleUNet,
    transform: transforms.Compose,
    mask_definitions: Sequence[MaskDefinition],
    device: torch.device,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Produce boolean masks for each pellet class.

    Despite the historical name, the function now works on both CPU and GPU.
    """

    pred_mask, rgb_image = predict_with_unet(image_path, model, transform, device)

    class_masks: List[np.ndarray] = []
    for definition in mask_definitions:
        class_masks.append(pred_mask == definition.index)

    return class_masks, rgb_image


def get_area_gpu(
    image_path: Path,
    model: SimpleUNet,
    transform: transforms.Compose,
    mask_definitions: Sequence[MaskDefinition],
    device: torch.device,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Measure the pixel area for every pellet mask."""

    class_masks, rgb_image = get_cluster_mask_gpu(image_path, model, transform, mask_definitions, device)

    cluster_areas = np.fromiter((int(mask.sum()) for mask in class_masks), dtype=np.int64)
    return cluster_areas, class_masks, rgb_image


def process_all_images(
    image_groups: Sequence[Sequence[Path]],
    model: SimpleUNet,
    transform: transforms.Compose,
    mask_definitions: Sequence[MaskDefinition],
    device: torch.device,
    *,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute cluster areas for every image discovered in the dataset.

    Returns:
        Array of shape (n_images, len(mask_definitions)) containing per-image pixel counts.
    """

    image_paths = [path for group in image_groups for path in group]
    if not image_paths:
        return np.zeros((0, len(mask_definitions)), dtype=np.int64)

    results: List[np.ndarray] = []
    iterator: Iterable[Path]

    if show_progress:
        iterator = tqdm(image_paths, desc="Extracting cluster areas")
    else:
        iterator = image_paths

    for image_path in iterator:
        try:
            cluster_areas, _, _ = get_area_gpu(image_path, model, transform, mask_definitions, device)
            results.append(cluster_areas)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing %s: %s", image_path, exc)
            results.append(np.zeros(len(mask_definitions), dtype=np.int64))

    return np.stack(results, axis=0)


def compute_approximate_npk(
    cluster_areas: Sequence[float],
    mask_definitions: Sequence[MaskDefinition] = DEFAULT_MASK_DEFINITIONS,
) -> np.ndarray:
    """Convert cluster areas into a rough NPK estimate using the nutrient table."""

    areas = np.asarray(cluster_areas, dtype=np.float64)
    total_pixels = areas.sum()

    if total_pixels <= 0:
        return np.zeros(len(NUTRIENT_KEYS), dtype=np.float64)

    totals = np.zeros(len(NUTRIENT_KEYS), dtype=np.float64)
    key_index = {key: idx for idx, key in enumerate(NUTRIENT_KEYS)}

    for area, definition in zip(areas, mask_definitions):
        if area <= 0:
            continue
        for key, weight in definition.composition.items():
            if key in key_index:
                totals[key_index[key]] += weight * area

    return totals / total_pixels


def get_npk(
    cluster_areas: Sequence[float],
    mask_definitions: Sequence[MaskDefinition] = DEFAULT_MASK_DEFINITIONS,
) -> np.ndarray:
    """
    Backwards-compatible wrapper around :func:`compute_approximate_npk`.

    Kept to mirror the notebook API while providing the improved implementation.
    """

    return compute_approximate_npk(cluster_areas, mask_definitions)


def print_stats(errors: np.ndarray, label: str) -> Tuple[float, float]:
    """Log MAE and RMSE statistics for a set of errors."""

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    logger.info("%s MAE: %.3f, RMSE: %.3f", label, mae, rmse)
    return mae, rmse


def plot_regression_diagnostics(
    approx_npk: np.ndarray,
    actual_npk: np.ndarray,
    predicted_npk: np.ndarray,
    *,
    components: Sequence[str] = NUTRIENT_KEYS,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Generate scatter/error plots comparing raw and regressed NPK predictions."""

    approx_arr = np.asarray(approx_npk)
    actual_arr = np.asarray(actual_npk)
    predicted_arr = np.asarray(predicted_npk)

    fig, axes = plt.subplots(3, len(components), figsize=(6 * len(components), 15))
    fig.suptitle("NPK Prediction Comparison: Raw Approximate vs Regression", fontsize=18, fontweight="bold")

    for idx, component in enumerate(components):
        color_cycle = ("blue", "red", "green")
        color = color_cycle[idx % len(color_cycle)]

        ax_actual_raw = axes[0, idx]
        ax_actual_reg = axes[1, idx]
        ax_errors = axes[2, idx]

        ax_actual_raw.scatter(
            actual_arr[:, idx],
            approx_arr[:, idx],
            alpha=0.6,
            color=color,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        min_val = min(np.min(actual_arr[:, idx]), np.min(approx_arr[:, idx]))
        max_val = max(np.max(actual_arr[:, idx]), np.max(approx_arr[:, idx]))
        ax_actual_raw.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)
        ax_actual_raw.set_xlabel(f"Actual {component}")
        ax_actual_raw.set_ylabel(f"Raw Approximate {component}")
        ax_actual_raw.set_title(f"{component}: Actual vs Raw Approximate")
        ax_actual_raw.grid(True, alpha=0.3)

        ax_actual_reg.scatter(
            actual_arr[:, idx],
            predicted_arr[:, idx],
            alpha=0.6,
            color=color,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        min_val = min(np.min(actual_arr[:, idx]), np.min(predicted_arr[:, idx]))
        max_val = max(np.max(actual_arr[:, idx]), np.max(predicted_arr[:, idx]))
        ax_actual_reg.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)
        ax_actual_reg.set_xlabel(f"Actual {component}")
        ax_actual_reg.set_ylabel(f"Regression Predicted {component}")
        ax_actual_reg.set_title(f"{component}: Actual vs Regression Predicted")
        ax_actual_reg.grid(True, alpha=0.3)

        raw_errors = approx_arr[:, idx] - actual_arr[:, idx]
        reg_errors = predicted_arr[:, idx] - actual_arr[:, idx]
        ax_errors.scatter(
            range(len(raw_errors)),
            raw_errors,
            label="Raw Error",
            alpha=0.7,
            color="gray",
            s=30,
        )
        ax_errors.scatter(
            range(len(reg_errors)),
            reg_errors,
            label="Regression Error",
            alpha=0.7,
            color=color,
            s=30,
        )
        ax_errors.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax_errors.set_xlabel("Sample Index")
        ax_errors.set_ylabel("Error")
        ax_errors.set_title(f"{component}: Prediction Errors Comparison")
        ax_errors.legend()
        ax_errors.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved regression diagnostics figure to %s", save_path)

    if show and not save_path:
        plt.show()
    else:
        plt.close(fig)


def summarize_regression_errors(
    actual_npk: np.ndarray,
    approx_npk: np.ndarray,
    predicted_npk: np.ndarray,
    *,
    components: Sequence[str] = NUTRIENT_KEYS,
) -> None:
    """Log MAE/RMSE for raw approximations and regression outputs."""

    actual_arr = np.asarray(actual_npk)
    approx_arr = np.asarray(approx_npk)
    predicted_arr = np.asarray(predicted_npk)

    logger.info("=== Error statistics comparison ===")
    for idx, component in enumerate(components):
        logger.info("Component: %s", component)
        raw_errors = approx_arr[:, idx] - actual_arr[:, idx]
        reg_errors = predicted_arr[:, idx] - actual_arr[:, idx]
        print_stats(raw_errors, "Raw Approximate")
        print_stats(reg_errors, "Regression Predicted")


def parse_actual_npk_from_path(image_path: Path) -> Optional[np.ndarray]:
    """Infer the ground-truth NPK value from the parent folder name."""

    folder_name = image_path.parent.name.replace(" ", "")
    parts = folder_name.split("-")
    if len(parts) != len(NUTRIENT_KEYS):
        return None

    try:
        values = [float(part) for part in parts]
    except ValueError:
        logger.debug("Unable to parse NPK from folder name: %s", folder_name)
        return None

    return np.asarray(values, dtype=np.float64)


def build_regression_samples(
    image_groups: Sequence[Sequence[Path]],
    model: SimpleUNet,
    transform: transforms.Compose,
    mask_definitions: Sequence[MaskDefinition],
    device: torch.device,
    *,
    show_progress: bool = True,
) -> List[RegressionSample]:
    """Extract regression features/labels from every image."""

    samples: List[RegressionSample] = []
    image_paths = [path for group in image_groups for path in group]
    iterator: Iterable[Path]

    if show_progress:
        iterator = tqdm(image_paths, desc="Building regression samples")
    else:
        iterator = image_paths

    for image_path in iterator:
        try:
            cluster_areas, _, _ = get_area_gpu(image_path, model, transform, mask_definitions, device)
            approx_npk = compute_approximate_npk(cluster_areas, mask_definitions)
            actual_npk = parse_actual_npk_from_path(image_path)
            samples.append(
                RegressionSample(
                    image_path=Path(image_path),
                    cluster_areas=cluster_areas,
                    approx_npk=approx_npk,
                    actual_npk=actual_npk,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to build sample for %s: %s", image_path, exc)

    return samples


def prepare_training_arrays(samples: Sequence[RegressionSample]) -> Tuple[np.ndarray, np.ndarray]:
    """Split samples into features and labels, discarding unlabeled entries."""

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for sample in samples:
        if sample.actual_npk is None:
            continue
        features.append(sample.approx_npk)
        labels.append(sample.actual_npk)

    if not features:
        raise ValueError("No labeled samples available for regression training.")

    return np.stack(features, axis=0), np.stack(labels, axis=0)


def tune_regressor(
    approx_npk: np.ndarray,
    actual_npk: np.ndarray,
    *,
    degree: int = 2,
    alpha: float = 1.0,
    save_path: Optional[Path] = None,
) -> Pipeline:
    """
    Train a polynomial-Ridge regression model and optionally persist it.

    Args:
        approx_npk: Array of shape (n_samples, 3) with approximate NPK values.
        actual_npk: Array of shape (n_samples, 3) with ground-truth labels.
        degree: Polynomial feature degree.
        alpha: Ridge regularisation strength.
        save_path: When provided, save the trained pipeline to this location.
    """

    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    pipeline.fit(approx_npk, actual_npk)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        logger.info("Saved regression model to %s", save_path)

    return pipeline


def load_npk_regressor(model_path: Path) -> Pipeline:
    """Load a previously trained regression pipeline."""

    model_path = Path(model_path)
    regressor: Pipeline = joblib.load(model_path)
    logger.info("Loaded regression model from %s", model_path)
    return regressor


def predict_npk(
    regressor: Pipeline,
    cluster_areas: Sequence[float],
    mask_definitions: Sequence[MaskDefinition] = DEFAULT_MASK_DEFINITIONS,
) -> np.ndarray:
    """Predict calibrated NPK values for a new image."""

    approx = compute_approximate_npk(cluster_areas, mask_definitions).reshape(1, -1)
    return regressor.predict(approx).flatten()


def run_regression_training(config: RegressionConfig) -> Pipeline:
    """High-level helper that orchestrates the full regression-training flow."""

    device = determine_device(config.device)
    checkpoint = config.checkpoint_path or resolve_checkpoint_path()
    model = load_unet_model(checkpoint, num_classes=config.num_classes, device=device)
    transform = build_unet_transform(config.image_size)

    logger.info("Collecting images from %s", config.dataset_dir)
    image_groups = auto_detect_image_folders(config.dataset_dir, image_extensions=config.image_extensions)
    if not image_groups:
        raise FileNotFoundError(f"No images found in {config.dataset_dir}")

    samples = build_regression_samples(
        image_groups,
        model,
        transform,
        config.mask_definitions,
        device,
        show_progress=config.show_progress,
    )

    features, labels = prepare_training_arrays(samples)
    logger.info("Training regression model on %d samples", features.shape[0])

    regressor = tune_regressor(
        features,
        labels,
        degree=config.polynomial_degree,
        alpha=config.ridge_alpha,
        save_path=config.regressor_output,
    )

    if config.generate_visualizations:
        predictions = regressor.predict(features)
        plot_regression_diagnostics(
            approx_npk=features,
            actual_npk=labels,
            predicted_npk=predictions,
            save_path=config.visualization_output,
            show=config.visualization_output is None,
        )
        summarize_regression_errors(labels, features, predictions)

    return regressor


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the NPK regression model.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/regressor_dataset"),
        help="Directory of labeled images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the segmentation checkpoint (defaults to checkpoints/best_model.pth).",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/regression_model.pkl"), help="Output path for the regression model.")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of classes expected by the segmentation model.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Resize dimension used for inference.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for regression features.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation strength.")
    parser.add_argument("--device", type=str, default=None, help="Torch device identifier (e.g., cuda, cpu).")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate diagnostic plots and error statistics after training.",
    )
    parser.add_argument(
        "--viz-output",
        type=Path,
        default=None,
        help="Optional path to save the visualization figure.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_arg_parser().parse_args(argv)

    config = RegressionConfig(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        regressor_output=args.output,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=args.device,
        polynomial_degree=args.degree,
        ridge_alpha=args.alpha,
        show_progress=not args.no_progress,
        generate_visualizations=args.visualize,
        visualization_output=args.viz_output,
    )

    run_regression_training(config)


if __name__ == "__main__":
    main()
