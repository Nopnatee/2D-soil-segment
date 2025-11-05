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
