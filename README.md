﻿**2D Soil Segment – src Layout**

- Source code lives under `src/soil_segment` using the standard src layout.
- All runnable examples live under `scripts/`.
- A single root wrapper `cli.py` provides unified commands.

**Install (editable)**

- Create/activate a virtual environment.
- Install dependencies Run: `pip install -e .`
- Install Pytorch for cuda 13.0 Run: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
- Manually install cuda 13.0 via Nvidia official website [https://developer.nvidia.com/cuda-toolkit]

**Project Structure**

- `src/soil_segment/` – importable package (models and helpers)
- `src/soil_segment/unet_trainer.py` - training module (exposes `main()`)
- `src/soil_segment/visualizer.py` - training history visualizer (with `main()`)
- `scripts/` - runnable tools (`annotator.py`, `npk_pixel_predictor.py`, `gradio_app.py`)
- `tests/` - lightweight regression checks (run with `pytest`)
- `cli.py` - single unified wrapper at the repository root
- `checkpoints/` - generated weights (ignored by git, recreate locally as needed)
- `datasets/` - local datasets (ignored by git; populate before training/inference)

**Usage**

- From code: `from soil_segment.custom_unet import SimpleUNet`.
- After `pip install -e .`, you can also use:
  - `soil-segment-train` - runs `soil_segment.unet_trainer:main`
  - `soil-segment-viz` - runs `soil_segment.visualizer:main`
  - `soil-segment-annotate` - runs the automated annotation helper
  - `soil-segment-predict` - runs the pixel-wise NPK predictor CLI
  - `soil-segment-gradio` - launches the Gradio-based NPK demo

**Unified CLI**

- `python cli.py train` - delegates to `soil_segment.unet_trainer:main`.
- `python cli.py viz checkpoints` – delegates to `soil_segment.visualizer:main`.
- `cli.py` is the only root-level wrapper.

**Uncoated Mode**

- Use uncoated mode to keep datasets, checkpoints, and model files separate from the default workflow.
- UNet uncoated dataset path: `datasets/UNET_dataset_uncoated`.
- Regression uncoated dataset path: `datasets/regression_dataset_uncoated`.
- Uncoated checkpoints/output path: `checkpoints_uncoated`.

**Uncoated Commands**

- Train UNet uncoated model:
  - `python cli.py train --uncoated`
  - Behavior in uncoated mode:
    - Leave-one-out cross-validation (5 folds for 5 images)
    - Train on 4 images, validate on 1 image, rotate folds
    - Random patch sampling (strong augmentation) instead of full-image-only training
    - Class-imbalance loss: weighted CrossEntropy + Dice
- Train regression model using uncoated dataset/checkpoint:
  - `python -m soil_segment.regression_trainer --uncoated`
  - Optional tuning flags for uncoated UNet:
    - `--uncoated-patch-size` (default `512`)
    - `--uncoated-patches-per-image` (default `64`)
    - `--uncoated-batch-size` (default `4`)
    - `--uncoated-epochs` (default `160`)
    - `--uncoated-patience` (default `30`)
    - `--uncoated-learning-rate` (default `1e-4`)
    - `--uncoated-weight-decay` (default `1e-4`)
    - `--uncoated-init-checkpoint` (warm-start source; default auto-uses `checkpoints/best_model.pth` when available)

**Uncoated Output Files**

- `checkpoints_uncoated/best_model_fold<F>_uncoated.pth`
- `checkpoints_uncoated/checkpoint_epoch_<N>_fold<F>_uncoated.pth`
- `checkpoints_uncoated/loo_cv_summary_uncoated.json` (includes per-fold and mean Dice)
- `checkpoints_uncoated/regression_model_uncoated.pkl`

**Overwrite Behavior**

- Running default mode updates files in `checkpoints/` only.
- Running `--uncoated` updates files in `checkpoints_uncoated/` only.
- `best_model.pth` and `best_model_uncoated.pth` are isolated from each other.
