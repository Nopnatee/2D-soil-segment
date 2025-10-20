"""Train the regression model that maps U-Net segmentations to NPK values."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from ..models.unet import SimpleUNet

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "assets" / "checkpoints" / "best_model.pth"
DEFAULT_OUTPUT_MODEL = ROOT_DIR / "assets" / "checkpoints" / "regression_model.pkl"


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------


@dataclass
class RegressionConfig:
    dataset_dir: Path = Path("regressor_dataset")
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    output_model: Path = DEFAULT_OUTPUT_MODEL
    num_classes: int = 5
    bead_masks: int = 4
    img_size: int = 256
    use_gpu: bool = True
    visualize: bool = True
    polynomial_degree: int = 2
    ridge_alpha: float = 1.0


@dataclass
class RegressionResult:
    cluster_areas: np.ndarray
    approx_npk: np.ndarray
    actual_npk: np.ndarray
    regressor: Ridge


# ---------------------------------------------------------------------------
# U-Net helpers
# ---------------------------------------------------------------------------


def load_unet_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> SimpleUNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleUNet(in_channels=3, n_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    return model


def build_unet_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_mask(
    image_path: Path,
    model: SimpleUNet,
    image_transform: transforms.Compose,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    input_tensor = image_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    if output.shape[1] == 1:
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    else:
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8),
        (rgb_image.shape[1], rgb_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return pred_mask, rgb_image


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------


def discover_image_folders(base_path: Path) -> List[List[Path]]:
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset folder '{base_path}' does not exist.")

    image_sets: List[List[Path]] = []
    subdirs = sorted([p for p in base_path.iterdir() if p.is_dir()])

    for subdir in subdirs:
        image_files = sorted(
            [p for p in subdir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        )
        if image_files:
            image_sets.append(image_files)
    if not image_sets:
        raise FileNotFoundError(f"No images found inside '{base_path}'.")
    return image_sets


def extract_cluster_masks(pred_mask: np.ndarray, bead_masks: int, num_classes: int) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    for class_id in range(1, min(num_classes, bead_masks + 1)):
        masks.append(pred_mask == class_id)
    while len(masks) < bead_masks:
        masks.append(np.zeros_like(pred_mask, dtype=bool))
    return masks[:bead_masks]


def measure_cluster_areas(masks: Iterable[np.ndarray], device: torch.device) -> List[int]:
    areas: List[int] = []
    use_gpu = device.type == "cuda"

    for mask in masks:
        if use_gpu:
            mask_tensor = torch.from_numpy(mask).to(device)
            area = int(torch.sum(mask_tensor).item())
        else:
            area = int(np.sum(mask))
        areas.append(area)

    return areas


def process_images(
    image_groups: List[List[Path]],
    model: SimpleUNet,
    transform: transforms.Compose,
    device: torch.device,
    bead_masks: int,
    num_classes: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    cluster_areas: List[List[int]] = []
    masks_cache: List[np.ndarray] = []

    total_images = sum(len(group) for group in image_groups)
    progress = tqdm(total=total_images, desc="Processing images")

    for group in image_groups:
        for image_path in group:
            pred_mask, _ = predict_mask(image_path, model, transform, device)
            masks_cache.append(pred_mask)

            masks = extract_cluster_masks(pred_mask, bead_masks, num_classes)
            areas = measure_cluster_areas(masks, device)
            cluster_areas.append(areas)

            progress.update(1)

    progress.close()
    return np.array(cluster_areas, dtype=np.int32), masks_cache


# ---------------------------------------------------------------------------
# NPK utilities
# ---------------------------------------------------------------------------


BEAD_COMPOSITIONS = [
    {"N": 18, "P": 46, "K": 0},  # black
    {"N": 0, "P": 0, "K": 60},  # red
    {"N": 21, "P": 0, "K": 0},  # stain
    {"N": 46, "P": 0, "K": 0},  # white
]


def decode_actual_npk(image_groups: List[List[Path]]) -> np.ndarray:
    npk_values: List[List[int]] = []

    for group in image_groups:
        for image_path in group:
            folder_name = image_path.parent.name
            try:
                npk = list(map(int, folder_name.split("-")))
                if len(npk) != 3:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"Folder '{folder_name}' does not encode an N-P-K triple.") from exc
            npk_values.append(npk)

    return np.array(npk_values, dtype=np.int32)


def approximate_npk_from_areas(areas: np.ndarray) -> np.ndarray:
    approx = []
    for area_vector in areas:
        totals = {"N": 0.0, "P": 0.0, "K": 0.0}
        total_pixels = area_vector.sum()
        if total_pixels == 0:
            approx.append([0.0, 0.0, 0.0])
            continue
        for idx, area in enumerate(area_vector):
            for key in totals:
                totals[key] += BEAD_COMPOSITIONS[idx][key] * area
        approx.append([round(totals[k] / total_pixels, 6) for k in ("N", "P", "K")])
    return np.array(approx, dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualisation and reporting
# ---------------------------------------------------------------------------


def plot_matching_analysis(actual: np.ndarray, approx: np.ndarray) -> None:
    components = ["N", "P", "K"]
    colours = ["blue", "red", "green"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("NPK Matching Analysis", fontsize=16, fontweight="bold")

    for i, (component, colour) in enumerate(zip(components, colours)):
        ax = axes[0, i]
        ax.scatter(actual[:, i], approx[:, i], alpha=0.6, color=colour, edgecolors="black", linewidth=0.5)
        min_val = min(actual[:, i].min(), approx[:, i].min())
        max_val = max(actual[:, i].max(), approx[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect match")
        ax.set_xlabel(f"Actual {component}")
        ax.set_ylabel(f"Approx {component}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        corr = np.corrcoef(actual[:, i], approx[:, i])[0, 1]
        ax.text(
            0.05,
            0.95,
            f"R = {corr:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

    errors = approx - actual
    ax = axes[1, 0]
    for i, (component, colour) in enumerate(zip(components, colours)):
        ax.scatter(range(len(errors)), errors[:, i], label=f"{component} error", alpha=0.7, color=colour, s=30)
    ax.axhline(y=0, color="black", alpha=0.5)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Prediction error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.hist([errors[:, i] for i in range(3)], bins=15, alpha=0.7, label=components, color=colours)
    ax.set_xlabel("Error value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.axis("off")
    rows = []
    for i, component in enumerate(components):
        mae = mean_absolute_error(actual[:, i], approx[:, i])
        rmse = mean_squared_error(actual[:, i], approx[:, i], squared=False)
        corr = np.corrcoef(actual[:, i], approx[:, i])[0, 1]
        rows.append([component, f"{mae:.3f}", f"{rmse:.3f}", f"{corr:.3f}"])
    table = ax.table(
        cellText=rows,
        colLabels=["Component", "MAE", "RMSE", "Correlation"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    plt.tight_layout()
    plt.show()


def print_summary(actual: np.ndarray, approx: np.ndarray, predicted: np.ndarray) -> None:
    components = ["N", "P", "K"]
    print("\nNPK analysis summary")
    print("=" * 50)

    for i, component in enumerate(components):
        mae_raw = mean_absolute_error(actual[:, i], approx[:, i])
        rmse_raw = mean_squared_error(actual[:, i], approx[:, i], squared=False)
        mae_reg = mean_absolute_error(actual[:, i], predicted[:, i])
        rmse_reg = mean_squared_error(actual[:, i], predicted[:, i], squared=False)
        print(f"{component}: raw MAE={mae_raw:.3f}, raw RMSE={rmse_raw:.3f}, "
              f"reg MAE={mae_reg:.3f}, reg RMSE={rmse_reg:.3f}")


# ---------------------------------------------------------------------------
# Regression model training
# ---------------------------------------------------------------------------


def train_regressor(
    approx_npk: np.ndarray,
    actual_npk: np.ndarray,
    *,
    degree: int,
    alpha: float,
    output_model: Path,
) -> Ridge:
    regressor = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
    regressor.fit(approx_npk, actual_npk)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, output_model)
    print(f"Saved regression model to {output_model}")
    return regressor


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_regression_pipeline(config: RegressionConfig) -> RegressionResult:
    device = torch.device("cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu")

    transform = build_unet_transform(config.img_size)
    model = load_unet_model(config.checkpoint_path, config.num_classes, device)
    image_groups = discover_image_folders(config.dataset_dir)

    cluster_areas, masks = process_images(
        image_groups=image_groups,
        model=model,
        transform=transform,
        device=device,
        bead_masks=config.bead_masks,
        num_classes=config.num_classes,
    )

    actual_npk = decode_actual_npk(image_groups)
    approx_npk = approximate_npk_from_areas(cluster_areas)

    if config.visualize and masks:
        plot_matching_analysis(actual_npk, approx_npk)

    regressor = train_regressor(
        approx_npk,
        actual_npk,
        degree=config.polynomial_degree,
        alpha=config.ridge_alpha,
        output_model=config.output_model,
    )

    predicted_npk = regressor.predict(approx_npk)
    print_summary(actual_npk, approx_npk, predicted_npk)

    return RegressionResult(
        cluster_areas=cluster_areas,
        approx_npk=approx_npk,
        actual_npk=actual_npk,
        regressor=regressor,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> RegressionConfig:
    parser = argparse.ArgumentParser(description="Train the NPK regression model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("regressor_dataset"), help="Directory containing bead image folders")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Path to trained U-Net checkpoint")
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL, help="Where to store the trained regressor")
    parser.add_argument("--num-classes", type=int, default=5, help="Total segmentation classes (including background)")
    parser.add_argument("--bead-masks", type=int, default=4, help="Number of bead clusters (excludes background)")
    parser.add_argument("--img-size", type=int, default=256, help="Image size expected by the U-Net")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU inference even if CUDA is available")
    parser.add_argument("--no-visualize", action="store_true", help="Skip plotting diagnostic figures")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial feature degree for the regression model")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation strength")

    args = parser.parse_args(argv)

    return RegressionConfig(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint_path,
        output_model=args.output_model,
        num_classes=args.num_classes,
        bead_masks=args.bead_masks,
        img_size=args.img_size,
        use_gpu=not args.no_gpu,
        visualize=not args.no_visualize,
        polynomial_degree=args.degree,
        ridge_alpha=args.alpha,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_regression_pipeline(config)


if __name__ == "__main__":
    main()
