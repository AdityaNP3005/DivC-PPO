from __future__ import annotations

from pathlib import Path

# Single source of truth for runtime paths and task identifiers.
# All paths are relative to this checkout.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
LOG_ROOT = f"{PROJECT_ROOT}/logs/sb3"
EVAL_ROOT = f"{PROJECT_ROOT}/eval_outputs"
ASSETS_ROOT = f"{PROJECT_ROOT}/assets"

USD_ROOT = f"{ASSETS_ROOT}/usd"

# Environment IDs and output namespace.
TRAIN_ENV_ID = "Isaac-UR10e-DivC-Reach-v0"
MODEL_DIR_NAME = "Isaac-UR10e-DivC-Reach-v0"

CUSTOM_SCENE_USD = f"{USD_ROOT}/stage_1.usd"


def log_base() -> str:
    return f"{LOG_ROOT}/{MODEL_DIR_NAME}"


def local_asset_path(path: str) -> str:
    """Map an asset path recorded in a checkpoint to this repository's assets/ folder.

    Checkpoints saved in another checkout or container keep their assets/... suffix;
    this repo's copy is used when the recorded path does not exist.
    """
    if "/assets/" in path and not Path(path).exists():
        local = Path(ASSETS_ROOT) / path.split("/assets/", 1)[1]
        if local.exists():
            return str(local)
    return path
