**2D Soil Segment — src Layout**

- Source code now lives under `src/soil_segment` using the standard src layout.
- Existing scripts (e.g., `app.py`, `app_ver2.py`, `script.py`) import from the package via `soil_segment.custom_unet`.

**Install (editable)**

- Create/activate a virtual environment.
- Run: `pip install -e .`

**Project Structure**

- `src/soil_segment/` — importable package (models and helpers)
- `app.py`, `app_ver2.py`, `script.py` — runnable scripts using the package
- `checkpoints/` — model weights and regressors
- `notebooks/` — Jupyter notebooks
- `UNET_dataset/`, `regressor_dataset/` — datasets

**Usage**

- From code: `from soil_segment.custom_unet import SimpleUNet`
- Scripts continue to work when run from the repo root. Ensure model files are present under `checkpoints/`.

**Notes**

- If you want entry points (CLI commands) for the apps, we can add them to `pyproject.toml`.
