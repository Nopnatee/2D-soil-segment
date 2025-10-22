**2D Soil Segment — src Layout**

- Source code lives under `src/soil_segment` using the standard src layout.
- Existing scripts (e.g., `app.py`, `app_ver2.py`, `script.py`) import from the package via `soil_segment.custom_unet`.

**Install (editable)**

- Create/activate a virtual environment.
- Run: `pip install -e .`

**Project Structure**

- `src/soil_segment/` — importable package (models and helpers)
- `src/soil_segment/trainer.py` — training module (with `main()`)
- `src/soil_segment/visualizer.py` — training history visualizer (with `main()`)
- `trainer.py`, `visualizer.py` — shims forwarding to the package
- `app.py`, `app_ver2.py`, `script.py` — runnable scripts using the package
- `checkpoints/` — model weights and regressors
- `notebooks/` — Jupyter notebooks
- `UNET_dataset/`, `regressor_dataset/` — datasets

**Usage**

- From code: `from soil_segment.custom_unet import SimpleUNet`
- Scripts continue to work when run from the repo root. Ensure model files are present under `checkpoints/`.
- After `pip install -e .`, you can also use:
  - `soil-segment-train` → runs `soil_segment.trainer:main`
  - `soil-segment-viz` → runs `soil_segment.visualizer:main`

**Notes**

- Legacy imports like `from custom_unet import SimpleUNet` still work via a small shim, but prefer `from soil_segment.custom_unet import SimpleUNet`.
