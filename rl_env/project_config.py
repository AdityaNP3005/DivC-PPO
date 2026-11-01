from __future__ import annotations

from pathlib import Path

# Single source of truth for runtime paths and task identifiers.
# Derive local paths on both host and container. Only shared IsaacLab/venv stay external.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
LOG_ROOT = f"{PROJECT_ROOT}/logs/sb3"
VIDEO_ROOT = f"{PROJECT_ROOT}/play_outputs"
ASSETS_ROOT = f"{PROJECT_ROOT}/assets"

STORAGE_ROOT = "/storage"
USD_ROOT = f"{ASSETS_ROOT}/usd"   # migrated into the repo (was /storage/usd)
ISAACLAB_ROOT = f"{STORAGE_ROOT}/IsaacLab"
VENV_SITE = f"{STORAGE_ROOT}/venv/lib/python3.11/site-packages"

# Environment IDs and output namespace.
TRAIN_ENV_ID = "Isaac-UR10e-DivC-Reach-v0"
MODEL_DIR_NAME = "Isaac-UR10e-DivC-Reach-v0"
PLAY_ENV_ID = "Isaac-UR10e-DivC-Reach-Play-v0"

CUSTOM_SCENE_USD = f"{USD_ROOT}/stage_1.usd"


def log_base() -> str:
    return f"{LOG_ROOT}/{MODEL_DIR_NAME}"


def local_asset_path(path: str) -> str:
    """Relocate this project's assets between its host and container mounts."""
    for root in (PROJECT_ROOT, "/divc_ppo"):
        prefix = f"{root}/assets/"
        if path.startswith(prefix):
            return str(Path(ASSETS_ROOT) / path[len(prefix):])
    return path
