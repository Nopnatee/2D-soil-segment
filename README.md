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
- `src/soil_segment/trainer.py` – training module (with `main()`)
- `src/soil_segment/visualizer.py` – training history visualizer (with `main()`)
- `scripts/` – standalone examples and apps (e.g. `npk_predictor.py`, `gradio_app.py`)
- `cli.py` – single unified wrapper at the repository root
- `checkpoints/` – model weights and regressors
- `datasets/UNET_dataset`, `datasets/regressor_dataset` – datasets

**Usage**

- From code: `from soil_segment.custom_unet import SimpleUNet`.
- After `pip install -e .`, you can also use:
  - `soil-segment-train` – runs `soil_segment.trainer:main`
  - `soil-segment-viz` – runs `soil_segment.visualizer:main`

**Unified CLI**

- `python cli.py train` – delegates to `soil_segment.trainer:main`.
- `python cli.py viz checkpoints` – delegates to `soil_segment.visualizer:main`.
- `cli.py` is the only root-level wrapper.
