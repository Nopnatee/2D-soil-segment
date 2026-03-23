"""
Training Server — FastAPI backend for the Vue.js training UI.

Place this file in the `scripts/` directory (or repo root).
Run with:  uvicorn training_server:app --reload --port 8000

Requires: pip install fastapi uvicorn python-multipart
(All other deps are already in pyproject.toml)
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path resolution — works whether placed in scripts/ or repo root
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_SCRIPTS_PARENT = _THIS.parent
# Repo root is the parent of scripts/ OR the file's own parent if at root
if _SCRIPTS_PARENT.name == "scripts":
    PROJECT_ROOT = _SCRIPTS_PARENT.parent
else:
    PROJECT_ROOT = _SCRIPTS_PARENT

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from soil_segment.config import get_data_paths  # type: ignore
    _PATHS = get_data_paths()
except Exception:
    _PATHS = {
        "unet_dataset": PROJECT_ROOT / "datasets" / "UNET_dataset",
        "regression_dataset": PROJECT_ROOT / "datasets" / "regression_dataset",
        "checkpoints": PROJECT_ROOT / "checkpoints",
    }

CLI_PATH = PROJECT_ROOT / "cli.py"
REGRESSION_MODULE = "soil_segment.regression_trainer"
TRAINING_UI_PATH = PROJECT_ROOT / "scripts" / "training_ui.html"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Soil Segment Training Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
@app.get("/training_ui.html", include_in_schema=False)
def training_ui():
    if not TRAINING_UI_PATH.is_file():
        raise HTTPException(404, f"Training UI not found: {TRAINING_UI_PATH}")
    return FileResponse(TRAINING_UI_PATH)

# ---------------------------------------------------------------------------
# Shared training state (single-process, one job at a time)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "running": False,
    "mode": None,          # "unet" | "unet_uncoated" | "regression"
    "metrics": [],         # list of epoch dicts
    "log": [],             # raw stdout lines (last 300)
    "process": None,       # subprocess.Popen
    "fold": None,          # current fold (uncoated mode)
    "total_folds": None,
    "stopped": False,
    "error": None,
    "completed": False,
}
_state_lock = threading.Lock()

# Regex patterns matching unet_trainer.py output
_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)"
    r"\s+\|\s+Train L:([\d.]+)\s+D:([\d.]+)\s+IoU:([\d.]+)"
    r"\s+\|\s+Val L:([\d.]+)\s+D:([\d.]+)\s+IoU:([\d.]+)"
    r"\s+\|\s+LR:([\d.e+-]+)"
    r".*?(\[BEST\])?"
)
_FOLD_RE = re.compile(r"Fold\s+(\d+)/(\d+)")
_EARLY_STOP_RE = re.compile(r"Early stopping triggered after (\d+) epochs")
_BEST_DICE_RE = re.compile(r"Best validation Dice:\s+([\d.]+)")
_REGRESSION_LINE_RE = re.compile(
    r"(?:R²|MAE|RMSE|MSE|score|Train|Val|Test).*?[-\d.]+", re.IGNORECASE
)

MAX_LOG_LINES = 400


# ---------------------------------------------------------------------------
# Background reader thread
# ---------------------------------------------------------------------------
def _stream_process(proc: subprocess.Popen, mode: str) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n\r")
        with _state_lock:
            _state["log"].append(line)
            if len(_state["log"]) > MAX_LOG_LINES:
                _state["log"] = _state["log"][-MAX_LOG_LINES:]

        # Parse fold header (uncoated)
        m_fold = _FOLD_RE.search(line)
        if m_fold:
            with _state_lock:
                _state["fold"] = int(m_fold.group(1))
                _state["total_folds"] = int(m_fold.group(2))

        # Parse epoch metrics
        m_ep = _EPOCH_RE.search(line)
        if m_ep:
            entry = {
                "epoch": int(m_ep.group(1)),
                "max_epoch": int(m_ep.group(2)),
                "train_loss": float(m_ep.group(3)),
                "train_dice": float(m_ep.group(4)),
                "train_iou": float(m_ep.group(5)),
                "val_loss": float(m_ep.group(6)),
                "val_dice": float(m_ep.group(7)),
                "val_iou": float(m_ep.group(8)),
                "lr": float(m_ep.group(9)),
                "best": bool(m_ep.group(10)),
                "fold": _state.get("fold"),
            }
            with _state_lock:
                _state["metrics"].append(entry)

    proc.wait()
    with _state_lock:
        _state["running"] = False
        _state["process"] = None
        _state["completed"] = (proc.returncode == 0)
        if proc.returncode not in (0, -9, -15):
            _state["error"] = f"Process exited with code {proc.returncode}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class UNetParams(BaseModel):
    uncoated: bool = False
    seed: int = 42
    # Default mode
    batch_size: int = 4
    img_size: int = 512
    epochs: int = 300
    patience: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Uncoated mode
    patch_size: int = 512
    patches_per_image: int = 64
    uncoated_batch_size: int = 4
    uncoated_epochs: int = 160
    uncoated_patience: int = 30
    uncoated_learning_rate: float = 1e-4
    uncoated_weight_decay: float = 1e-4
    uncoated_init_checkpoint: Optional[str] = None


class RegressionParams(BaseModel):
    uncoated: bool = False
    img_size: int = 512
    degree: int = 1
    alpha: float = 1.0
    no_ridge: bool = False
    no_scaling: bool = False
    test_size: float = 0.2
    seed: int = 42
    max_images: Optional[int] = None
    cpu: bool = False


# ---------------------------------------------------------------------------
# Helper: build CLI argv list
# ---------------------------------------------------------------------------
def _build_unet_cmd(p: UNetParams) -> List[str]:
    cmd = [sys.executable, str(CLI_PATH), "train", "--seed", str(p.seed)]
    if p.uncoated:
        cmd += [
            "--uncoated",
            "--uncoated-patch-size", str(p.patch_size),
            "--uncoated-patches-per-image", str(p.patches_per_image),
            "--uncoated-batch-size", str(p.uncoated_batch_size),
            "--uncoated-epochs", str(p.uncoated_epochs),
            "--uncoated-patience", str(p.uncoated_patience),
            "--uncoated-learning-rate", str(p.uncoated_learning_rate),
            "--uncoated-weight-decay", str(p.uncoated_weight_decay),
        ]
        if p.uncoated_init_checkpoint:
            cmd += ["--uncoated-init-checkpoint", p.uncoated_init_checkpoint]
    return cmd


def _build_regression_cmd(p: RegressionParams) -> List[str]:
    cmd = [
        sys.executable, "-m", REGRESSION_MODULE,
        "--img-size", str(p.img_size),
        "--degree", str(p.degree),
        "--alpha", str(p.alpha),
        "--test-size", str(p.test_size),
        "--seed", str(p.seed),
    ]
    if p.uncoated:
        cmd.append("--uncoated")
    if p.no_ridge:
        cmd.append("--no-ridge")
    if p.no_scaling:
        cmd.append("--no-scaling")
    if p.cpu:
        cmd.append("--cpu")
    if p.max_images is not None:
        cmd += ["--max-images", str(p.max_images)]
    return cmd


# ---------------------------------------------------------------------------
# Routes — Dataset
# ---------------------------------------------------------------------------
def _count_dir(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for f in path.iterdir() if f.is_file())


@app.get("/api/dataset/info")
def dataset_info():
    """Return file counts for all dataset directories."""
    def _info(base: Path):
        return {
            "base": str(base),
            "images": _count_dir(base / "images"),
            "masks": _count_dir(base / "masks"),
            "exists": base.exists(),
        }

    return {
        "unet": _info(Path(_PATHS["unet_dataset"])),
        "unet_uncoated": _info(Path(str(_PATHS["unet_dataset"]) + "_uncoated")),
        "regression": _info(Path(_PATHS["regression_dataset"])),
        "regression_uncoated": _info(Path(str(_PATHS["regression_dataset"]) + "_uncoated")),
        "checkpoints": {
            "base": str(_PATHS["checkpoints"]),
            "files": [
                f.name for f in Path(_PATHS["checkpoints"]).iterdir()
                if f.is_file()
            ] if Path(_PATHS["checkpoints"]).exists() else [],
        },
    }


@app.post("/api/dataset/upload")
async def upload_dataset(
    files: List[UploadFile] = File(...),
    dataset_type: str = "unet",         # unet | unet_uncoated | regression | regression_uncoated
    file_role: str = "images",          # images | masks
):
    """
    Upload one or more image/mask files (or a single .zip containing images/ and masks/).
    """
    suffix = "_uncoated" if "uncoated" in dataset_type else ""
    if "regression" in dataset_type:
        base = Path(str(_PATHS["regression_dataset"]) + suffix)
    else:
        base = Path(str(_PATHS["unet_dataset"]) + suffix)

    target_dir = base / file_role
    target_dir.mkdir(parents=True, exist_ok=True)

    saved, skipped = [], []

    for upload in files:
        data = await upload.read()
        fname = upload.filename or "file"

        # Handle zip upload — extract images/ and masks/ sub-dirs
        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for member in zf.namelist():
                        parts = Path(member).parts
                        if len(parts) >= 2 and parts[0] in ("images", "masks"):
                            dest = base / parts[0] / Path(*parts[1:])
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(zf.read(member))
                            saved.append(str(dest.relative_to(base)))
            except Exception as exc:
                raise HTTPException(400, f"Bad zip: {exc}")
        else:
            dest = target_dir / fname
            dest.write_bytes(data)
            saved.append(fname)

    return {"saved": saved, "skipped": skipped, "target": str(target_dir)}


# ---------------------------------------------------------------------------
# Routes — Training
# ---------------------------------------------------------------------------
@app.get("/api/train/status")
def train_status():
    with _state_lock:
        return {
            "running": _state["running"],
            "mode": _state["mode"],
            "fold": _state["fold"],
            "total_folds": _state["total_folds"],
            "metrics": list(_state["metrics"]),
            "log": list(_state["log"]),
            "completed": _state["completed"],
            "error": _state["error"],
        }


@app.post("/api/train/unet/start")
def start_unet(params: UNetParams):
    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "Training already running")
        _state.update({
            "running": True,
            "mode": "unet_uncoated" if params.uncoated else "unet",
            "metrics": [],
            "log": [],
            "fold": None,
            "total_folds": None,
            "stopped": False,
            "error": None,
            "completed": False,
        })

    cmd = _build_unet_cmd(params)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    with _state_lock:
        _state["process"] = proc

    t = threading.Thread(target=_stream_process, args=(proc, "unet"), daemon=True)
    t.start()
    return {"started": True, "cmd": " ".join(cmd)}


@app.post("/api/train/regression/start")
def start_regression(params: RegressionParams):
    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "Training already running")
        _state.update({
            "running": True,
            "mode": "regression",
            "metrics": [],
            "log": [],
            "fold": None,
            "total_folds": None,
            "stopped": False,
            "error": None,
            "completed": False,
        })

    cmd = _build_regression_cmd(params)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    with _state_lock:
        _state["process"] = proc

    t = threading.Thread(target=_stream_process, args=(proc, "regression"), daemon=True)
    t.start()
    return {"started": True, "cmd": " ".join(cmd)}


@app.post("/api/train/stop")
def stop_training():
    with _state_lock:
        proc = _state.get("process")
        if proc is None or not _state["running"]:
            raise HTTPException(400, "No training process is running")
        proc.terminate()
        _state["running"] = False
        _state["stopped"] = True
        _state["process"] = None
    return {"stopped": True}


@app.delete("/api/train/clear")
def clear_metrics():
    """Reset metrics and log without stopping a run."""
    with _state_lock:
        _state["metrics"] = []
        _state["log"] = []
        _state["error"] = None
        _state["completed"] = False
    return {"cleared": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("training_server:app", host="0.0.0.0", port=8000, reload=True)
