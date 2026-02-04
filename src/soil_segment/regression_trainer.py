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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
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


try:
    from scripts.npk_pixel_predictor import ClassStat, calculate_nutrient_breakdown
except ModuleNotFoundError:
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

    NUTRIENT_KEYS: Tuple[str, ...] = ("N", "P", "K", "S", "Mg", "Br", "Ca")

    RAW_MATERIAL_NUTRIENTS: Dict[str, Dict[str, float]] = {
        "urea": {"N": 46.0, "P": 0.0, "K": 0.0, "S": 0.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
        "dap": {"N": 18.0, "P": 45.5, "K": 0.0, "S": 0.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
        "mop": {"N": 0.0, "P": 0.0, "K": 60.0, "S": 0.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
        "ammoniumsulphate": {"N": 20.5, "P": 0.0, "K": 0.0, "S": 23.0, "Mg": 0.0, "Br": 0.0, "Ca": 0.0},
        "mg": {"N": 0.0, "P": 0.0, "K": 0.0, "S": 14.0, "Mg": 10.0, "Br": 0.0, "Ca": 0.0},
        "br": {"N": 0.0, "P": 0.0, "K": 0.0, "S": 0.0, "Mg": 0.0, "Br": 14.0, "Ca": 16.0},
    }

    CLASS_NAME_TO_RAW_MATERIAL: Dict[str, str] = {
        "yellowurea": "urea",
        "yellowureacoated": "urea",
        "yellowureauncoated": "urea",
        "urea": "urea",
        "blackdap": "dap",
        "dap": "dap",
        "redmop": "mop",
        "mop": "mop",
        "whiteamp": "ammoniumsulphate",
        "whiteams": "ammoniumsulphate",
        "whiteammoniumsulphate": "ammoniumsulphate",
        "whiteammoniumsulfate": "ammoniumsulphate",
        "ammoniumsulphate": "ammoniumsulphate",
        "ammoniumsulfate": "ammoniumsulphate",
        "amp": "ammoniumsulphate",
        "whiteboron": "br",
        "boron": "br",
        "br": "br",
        "whitemg": "mg",
        "mg": "mg",
    }

    def _normalize_class_name(name: str) -> str:
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def calculate_nutrient_breakdown(stats: Sequence[ClassStat]) -> NutrientBreakdown:
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

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_BEAD_MASKS = 6


def resolve_checkpoint_path(path: Optional[Path]) -> Path:
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
    raise FileNotFoundError(f"Could not locate a checkpoint. Tried: {tried}")


def _infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith("final_conv.weight"):
            return int(value.shape[0])
    raise KeyError("Checkpoint is missing final_conv.weight; cannot infer number of classes.")


def load_unet_model(checkpoint_path: Path, device: str) -> Tuple[SimpleUNet, int]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes = _infer_num_classes(state_dict)

    model = SimpleUNet(in_channels=3, n_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, num_classes


def build_unet_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def load_img_as_rgb(img_path: str) -> np.ndarray:
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_with_unet(
    image_path: str,
    model: SimpleUNet,
    transform: transforms.Compose,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_image = load_img_as_rgb(image_path)
    pil_image = Image.fromarray(rgb_image.astype(np.uint8))
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    if output.shape[1] == 1:
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    else:
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    original_size = rgb_image.shape[:2]
    pred_mask_resized = cv2.resize(
        pred_mask.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return pred_mask_resized, rgb_image


def get_cluster_masks(
    pred_mask: np.ndarray,
    bead_masks: int,
    num_classes: int,
) -> List[np.ndarray]:
    all_masks: List[np.ndarray] = []
    for class_id in range(1, min(num_classes, bead_masks + 1)):
        class_mask = pred_mask == class_id
        all_masks.append(class_mask)

    while len(all_masks) < bead_masks:
        all_masks.append(np.zeros(pred_mask.shape, dtype=bool))

    return all_masks[:bead_masks]


def get_area_counts(
    pred_mask: np.ndarray,
    bead_masks: int,
    num_classes: int,
) -> List[int]:
    masks = get_cluster_masks(pred_mask, bead_masks, num_classes)
    return [int(np.count_nonzero(mask)) for mask in masks]


def process_all_images(
    img_paths: Iterable[str],
    model: SimpleUNet,
    transform: transforms.Compose,
    device: str,
    *,
    bead_masks: int,
    num_classes: int,
) -> np.ndarray:
    results: List[List[int]] = []
    for img_path in tqdm(list(img_paths), desc="Processing images"):
        pred_mask, _ = predict_with_unet(img_path, model, transform, device)
        cluster_areas = get_area_counts(pred_mask, bead_masks, num_classes)
        results.append(cluster_areas)
    return np.array(results, dtype=np.int64)


def load_class_names(num_classes: int, class_json: Optional[Path] = None) -> List[str]:
    default_candidates = [
        PROJECT_ROOT / "datasets" / "UNET_dataset" / "classes.json",
        PROJECT_ROOT / "datasets" / "annotate" / "classes.json",
    ]
    class_file = class_json or next((p for p in default_candidates if p.exists()), default_candidates[0])
    names: List[str] = []

    if class_file.exists():
        try:
            data = json.loads(class_file.read_text())
            names = data.get("class_names", []) or []
        except json.JSONDecodeError:
            names = []

    if not names:
        names = list(DEFAULT_CLASS_NAMES)

    if len(names) < num_classes:
        names.extend(f"Class {idx}" for idx in range(len(names), num_classes))

    return names[:num_classes]


def build_class_stats(mask: np.ndarray, class_names: Sequence[str]) -> List[ClassStat]:
    total_pixels = mask.size
    stats: List[ClassStat] = []
    for idx, name in enumerate(class_names):
        pixel_count = int(np.sum(mask == idx))
        percentage = (pixel_count / total_pixels) * 100 if total_pixels else 0.0
        stats.append(ClassStat(idx=idx, name=name, pixel_count=pixel_count, percentage=percentage))
    return stats


def get_npk(cluster_areas: Sequence[int], class_names: Optional[Sequence[str]] = None) -> List[float]:
    class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
    background_name = class_names[0] if class_names else "background"
    stats = [ClassStat(idx=0, name=background_name, pixel_count=0, percentage=0.0)]
    for idx, area in enumerate(cluster_areas, start=1):
        name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        try:
            area_value = float(area)
        except (TypeError, ValueError):
            area_value = 0.0
        if not np.isfinite(area_value) or area_value < 0:
            area_value = 0.0
        stats.append(
            ClassStat(
                idx=idx,
                name=name,
                pixel_count=int(area_value),
                percentage=0.0,
            )
        )
    breakdown = calculate_nutrient_breakdown(stats)
    return [round(float(breakdown.totals[key]), 6) for key in ("N", "P", "K")]


def _parse_npk_from_path(path_str: str) -> List[int]:
    p = Path(path_str)
    for candidate in (p.parent, p.parent.parent):
        parts = candidate.name.split("-")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return list(map(int, parts))
    raise ValueError(f"Could not parse NPK from path: {path_str}")


def _collect_image_paths(dataset_dir: Path) -> List[List[str]]:
    img_path_list: List[List[str]] = []
    subdirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]

    for subdir in subdirs:
        image_files = [p for p in sorted(subdir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

        if not image_files:
            images_dir = subdir / "images"
            if images_dir.is_dir():
                image_files = [
                    p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS
                ]

        if image_files:
            img_path_list.append([str(p) for p in image_files])

    return img_path_list


def auto_detect_image_folders(base_path: Optional[Path]) -> Tuple[List[List[str]], Optional[Path]]:
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
        else:
            _register_candidate(Path.cwd() / base_path.name)
            _register_candidate(PROJECT_ROOT / base_path.name)
            _register_candidate(PROJECT_ROOT / "datasets" / base_path.name)

        if base_path.name == "regression_dataset":
            _register_candidate(PROJECT_ROOT / "datasets" / "regressor_dataset")

    resolved_base: Optional[Path] = None
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_dir():
            img_path_list = _collect_image_paths(candidate)
            if img_path_list:
                resolved_base = candidate.resolve()
                break

    return img_path_list, resolved_base


def _prepare_training_data(
    approx_features: Sequence[Sequence[float]],
    actual_targets: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    approx_arr = np.asarray(approx_features, dtype=np.float32)
    actual_arr = np.asarray(actual_targets, dtype=np.float32)
    if approx_arr.shape != actual_arr.shape:
        raise ValueError(
            f"approx_npk shape {approx_arr.shape} does not match actual_npk shape {actual_arr.shape}."
        )
    if approx_arr.ndim != 2:
        raise ValueError("Expected 2D arrays shaped as (n_samples, n_features).")
    return approx_arr, actual_arr


def build_npk_regressor(degree: int = 2, alpha: float = 1.0):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )


def tune_regressor(
    y_approx: Sequence[Sequence[float]],
    y_true: Sequence[Sequence[float]],
    save_path: Optional[Path] = None,
    degree: int = 2,
    alpha: float = 1.0,
):
    features, targets = _prepare_training_data(y_approx, y_true)
    regressor = build_npk_regressor(degree=degree, alpha=alpha)
    regressor.fit(features, targets)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(regressor, save_path)
        print(f"Model checkpoint saved to: {save_path}")

    return regressor


def load_npk_regressor(model_path: Path = Path("checkpoints/regression_model.pkl")):
    model_path = Path(model_path)
    try:
        regressor = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return regressor
    except FileNotFoundError:
        print(f"Model file not found at: {model_path}")
    except Exception as exc:
        print(f"Error loading model: {exc}")
    return None


def predict_npk(regressor, nutrient_features: Sequence[float]) -> np.ndarray:
    if regressor is None:
        raise ValueError("Regressor is not loaded.")
    features = np.asarray(nutrient_features, dtype=np.float32).reshape(1, -1)
    return regressor.predict(features).flatten()


def train_regressor_with_holdout(
    y_approx: Sequence[Sequence[float]],
    y_true: Sequence[Sequence[float]],
    *,
    save_path: Optional[Path] = Path("checkpoints/regression_model.pkl"),
    degree: int = 1,
    alpha: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
):
    features, targets = _prepare_training_data(y_approx, y_true)
    if len(features) < 2:
        raise ValueError("Need at least two samples to perform a validation split.")

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    validation_model = build_npk_regressor(degree=degree, alpha=alpha)
    validation_model.fit(x_train, y_train)
    val_predictions = validation_model.predict(x_val)

    val_mae = mean_absolute_error(y_val, val_predictions, multioutput="raw_values")
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions, multioutput="raw_values"))

    final_model = build_npk_regressor(degree=degree, alpha=alpha)
    final_model.fit(features, targets)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, save_path)
        print(f"Model checkpoint saved to: {save_path}")

    metrics = {
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_sample_count": len(x_val),
    }
    return final_model, metrics, (val_predictions, y_val)


def run_regression_training(
    approx_npk: Sequence[Sequence[float]],
    actual_npk: Sequence[Sequence[float]],
    **kwargs,
):
    regressor, metrics, holdout_results = train_regressor_with_holdout(
        approx_npk,
        actual_npk,
        **kwargs,
    )

    val_predictions, val_actuals = holdout_results
    nutrient_labels = ["N", "P", "K"] if val_predictions.shape[1] == 3 else [
        f"feature_{idx}" for idx in range(val_predictions.shape[1])
    ]

    print("=== Validation metrics ===")
    for label, mae, rmse in zip(nutrient_labels, metrics["val_mae"], metrics["val_rmse"]):
        print(f"{label}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    print(f"Validation samples: {metrics['val_sample_count']}")
    print("=== Sample validation predictions ===")
    preview_count = min(5, len(val_actuals))
    for idx in range(preview_count):
        pred_vec = np.round(val_predictions[idx], 4)
        actual_vec = np.round(val_actuals[idx], 4)
        print(f"{idx + 1:03d} | Predicted: {pred_vec.tolist()} | Actual: {actual_vec.tolist()}")

    return regressor, metrics, holdout_results


def summarize_regression_outputs(
    approx_npk: np.ndarray,
    actual_npk: np.ndarray,
    predicted_npk: np.ndarray,
    *,
    sample_limit: int = 10,
) -> None:
    components = ["N", "P", "K"]

    print("=== Error statistics comparison ===")
    for i, component in enumerate(components):
        raw_errors = approx_npk[:, i] - actual_npk[:, i]
        reg_errors = predicted_npk[:, i] - actual_npk[:, i]
        raw_mae = float(np.mean(np.abs(raw_errors)))
        raw_rmse = float(np.sqrt(np.mean(raw_errors ** 2)))
        reg_mae = float(np.mean(np.abs(reg_errors)))
        reg_rmse = float(np.sqrt(np.mean(reg_errors ** 2)))
        print(f"{component} raw MAE: {raw_mae:.3f}, RMSE: {raw_rmse:.3f}")
        print(f"{component} reg MAE: {reg_mae:.3f}, RMSE: {reg_rmse:.3f}")

    print("Detailed per-sample comparison (raw vs regression vs actual):")
    preview_count = min(sample_limit, len(actual_npk))
    for i in range(preview_count):
        raw_vec = np.round(approx_npk[i], 4)
        reg_vec = np.round(predicted_npk[i], 4)
        act_vec = np.round(actual_npk[i], 4)
        print(f"{i + 1:03d} | raw {raw_vec.tolist()} | reg {reg_vec.tolist()} | actual {act_vec.tolist()}")

    if len(actual_npk) > preview_count:
        print(f"... and {len(actual_npk) - preview_count} more samples")


def prepare_regression_data(
    img_path_list: List[List[str]],
    model: SimpleUNet,
    transform: transforms.Compose,
    device: str,
    class_names: Sequence[str],
    *,
    max_images: Optional[int] = None,
    skip_errors: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    approx_npk: List[List[float]] = []
    actual_npk: List[List[int]] = []
    used_paths: List[str] = []

    flattened = [path for paths in img_path_list for path in paths]
    if max_images is not None:
        flattened = flattened[:max_images]

    for path in tqdm(flattened, desc="Computing NPK features"):
        try:
            pred_mask, _ = predict_with_unet(path, model, transform, device)
            stats = build_class_stats(pred_mask, class_names)
            breakdown = calculate_nutrient_breakdown(stats)
            approx_npk.append([breakdown.totals["N"], breakdown.totals["P"], breakdown.totals["K"]])
            actual_npk.append(_parse_npk_from_path(path))
            used_paths.append(path)
        except Exception as exc:
            print(f"[regression-data] Skipping {path}: {exc}")
            if not skip_errors:
                raise

    if not approx_npk:
        raise RuntimeError("No regression samples were prepared; check dataset paths and labels.")

    return (
        np.asarray(approx_npk, dtype=np.float32),
        np.asarray(actual_npk, dtype=np.float32),
        used_paths,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NPK regression model from segmented images.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the regression dataset directory (defaults to config lookup).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the U-Net checkpoint (.pth). Defaults to checkpoints/best_model.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save regression_model.pkl (defaults to checkpoints dir).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="Resize dimension for U-Net inference.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Polynomial degree for the regression model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out validation size (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the validation split.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to process (for quick checks).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast if any image fails during preprocessing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_path_list, dataset_root = auto_detect_image_folders(args.dataset)
    if not img_path_list or dataset_root is None:
        print("No valid image folders found for regression training.")
        return 1

    print(f"Using dataset directory: {dataset_root}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, num_classes = load_unet_model(checkpoint_path, device)
    print(f"Loaded U-Net with {num_classes} classes from {checkpoint_path}")

    class_names = load_class_names(num_classes)
    transform = build_unet_transform(args.img_size)

    approx_npk, actual_npk, _ = prepare_regression_data(
        img_path_list,
        model,
        transform,
        device,
        class_names,
        max_images=args.max_images,
        skip_errors=not args.strict,
    )

    output_path = args.output
    if output_path is None:
        output_path = get_data_paths()["checkpoints"] / "regression_model.pkl"

    regressor, _, _ = run_regression_training(
        approx_npk,
        actual_npk,
        save_path=output_path,
        degree=args.degree,
        test_size=args.test_size,
        random_state=args.seed,
    )

    predicted_npk = regressor.predict(approx_npk)
    summarize_regression_outputs(approx_npk, actual_npk, predicted_npk)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
