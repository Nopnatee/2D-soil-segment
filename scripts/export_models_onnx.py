"""Export the U-Net and regression models to ONNX for faster inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import torch

# Make sure the project sources are importable without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soil_segment.config import get_data_paths  # type: ignore
from soil_segment.custom_unet import SimpleUNet  # type: ignore


def _load_checkpoint_state(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(ckpt_path), map_location=device)
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return payload["model_state_dict"]
        if "state_dict" in payload:
            return payload["state_dict"]
        return payload
    if hasattr(payload, "state_dict"):
        return payload.state_dict()
    raise ValueError(f"Unsupported checkpoint format: {type(payload)}")


def _infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith("final_conv.weight"):
            return int(value.shape[0])
    raise KeyError("Checkpoint is missing final_conv.weight; cannot infer number of classes.")


def _resolve_device(device_name: str) -> torch.device:
    device_name = device_name.lower()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_name)


def _ensure_onnx_available() -> None:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        pip_cmd = f"{sys.executable} -m pip install onnx"
        raise ModuleNotFoundError(
            "onnx is required for export. "
            f"Install with `{pip_cmd}` (current interpreter: {sys.executable})."
        ) from exc


def export_unet_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    img_size: int,
    device: torch.device,
    opset: int,
    dynamic_axes: bool,
    force: bool,
) -> Path:
    _ensure_onnx_available()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"U-Net checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    state_dict = _load_checkpoint_state(checkpoint_path, device)
    num_classes = _infer_num_classes(state_dict)

    model = SimpleUNet(in_channels=3, n_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    input_names = ["input"]
    output_names = ["logits"]
    axes = None
    if dynamic_axes:
        axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    return output_path


def _infer_regression_features(model, fallback: Optional[int]) -> int:
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            feature_count = getattr(step, "n_features_in_", None)
            if feature_count is not None:
                break
    if feature_count is None:
        feature_count = fallback
    if feature_count is None:
        raise ValueError("Could not infer regression input feature count; use --regression-features.")
    return int(feature_count)


def export_regression_onnx(
    model_path: Path,
    output_path: Path,
    *,
    feature_count: Optional[int],
    force: bool,
) -> Path:
    _ensure_onnx_available()
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ModuleNotFoundError as exc:
        pip_cmd = f"{sys.executable} -m pip install skl2onnx"
        raise ModuleNotFoundError(
            "skl2onnx is required to export the regression model. "
            f"Install with `{pip_cmd}` (current interpreter: {sys.executable})."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Regression model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    model = joblib.load(model_path)
    input_features = _infer_regression_features(model, feature_count)
    initial_types = [("input", FloatTensorType([None, input_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types)

    from onnx import save_model

    save_model(onnx_model, str(output_path))
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    data_paths = get_data_paths()
    default_unet = data_paths["checkpoints"] / "best_model.pth"
    default_reg = data_paths["checkpoints"] / "regression_model.pkl"
    default_unet_onnx = data_paths["checkpoints"] / "best_model.onnx"
    default_reg_onnx = data_paths["checkpoints"] / "regression_model.onnx"

    parser = argparse.ArgumentParser(
        description="Export the U-Net and regression models to ONNX format."
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=default_unet,
        help=f"Path to the U-Net checkpoint (.pth). Defaults to {default_unet}.",
    )
    parser.add_argument(
        "--regression-model",
        type=Path,
        default=default_reg,
        help=f"Path to the regression model (.pkl). Defaults to {default_reg}.",
    )
    parser.add_argument(
        "--unet-onnx",
        type=Path,
        default=default_unet_onnx,
        help=f"Destination for the U-Net ONNX file. Defaults to {default_unet_onnx}.",
    )
    parser.add_argument(
        "--regression-onnx",
        type=Path,
        default=default_reg_onnx,
        help=f"Destination for the regression ONNX file. Defaults to {default_reg_onnx}.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="Square image size for the U-Net export dummy input (match training/inference).",
    )
    parser.add_argument(
        "--regression-features",
        type=int,
        default=None,
        help="Override input feature count for regression export (defaults to inferred).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to target.",
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Enable dynamic batch/height/width for the U-Net ONNX graph.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for U-Net export (cpu or cuda).",
    )
    parser.add_argument(
        "--skip-unet",
        action="store_true",
        help="Skip exporting the U-Net model.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip exporting the regression model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=True,
        help="Overwrite existing ONNX files (default: True).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        device = _resolve_device(args.device)
    except Exception as exc:
        print(f"[export-onnx] {exc}")
        return 1

    if args.skip_unet and args.skip_regression:
        print("[export-onnx] Nothing to export (both models skipped).")
        return 0

    if not args.skip_unet:
        try:
            unet_path = export_unet_onnx(
                args.unet_checkpoint,
                args.unet_onnx,
                img_size=args.img_size,
                device=device,
                opset=args.opset,
                dynamic_axes=args.dynamic_axes,
                force=args.force,
            )
            print(f"[export-onnx] U-Net exported to: {unet_path}")
        except Exception as exc:
            print(f"[export-onnx] U-Net export failed: {exc}")
            return 1

    if not args.skip_regression:
        try:
            reg_path = export_regression_onnx(
                args.regression_model,
                args.regression_onnx,
                feature_count=args.regression_features,
                force=args.force,
            )
            print(f"[export-onnx] Regression model exported to: {reg_path}")
        except Exception as exc:
            print(f"[export-onnx] Regression export failed: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
