"""Minimum-sample contract for deterministic inference."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer (0 = auto)")
    return number


def completion_status(requested, completed, steps, max_steps):
    complete = completed >= requested
    return {
        "validation_complete": complete,
        "episodes_requested": requested,
        "episodes_completed": completed,
        "steps_run": steps,
        "max_steps": max_steps,
        "exit_code": 0 if complete else 2,
        "reason": "minimum_met" if complete else "minimum_not_met",
    }


def file_digest(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".status-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        os.chmod(temporary, 0o664)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def evaluation_context(checkpoint, seed, num_envs):
    """Record the exact model, normalizer statistics and evaluation condition."""
    from evaluation.checkpoint_assets import resolve_vecnormalize_path
    from evaluation.provenance import checkpoint_identity, source_identity, runtime_assets
    checkpoint = str(Path(checkpoint).resolve())
    stats = resolve_vecnormalize_path(checkpoint)
    return {
        'schema_version': 2,
        **checkpoint_identity(checkpoint),
        'source': source_identity(),
        'runtime_assets': runtime_assets(),
        'evaluation_seed': int(seed), 'deterministic': True,
        "checkpoint": checkpoint, "checkpoint_sha256": file_digest(checkpoint),
        "normalizer": stats, "normalizer_sha256": file_digest(stats),
        "seed": int(seed), "num_envs": int(num_envs),
        "condition": os.environ.get("ABLATION_EVAL_CONDITION") or "nominal",
        "overrides": {key: os.environ.get(key, "") for key in (
            "YOLO_OBS_NOISE_STD", "PLAY_OBJECT_RADIUS_M", "POSITION_SCALE_M",
            "OBJECT_USD", "OBJECT_USD_SCALE", "OBJECT_CYL_HEIGHT_M",
            "CYLINDER_FLOOR_Z_M", "YOLO_MODEL_PATH", "REWARD_ERROR_SOURCE",
            "YOLO_CONF", "YOLO_IOU", "YOLO_DEVICE", "YOLO_MAX_DET", "YOLO_ROI_ENABLED")},
    }


def require_unchanged_assets(context):
    for path, digest in (("checkpoint", "checkpoint_sha256"), ("normalizer", "normalizer_sha256")):
        if file_digest(context[path]) != context[digest]:
            raise ValueError(f"Evaluation asset changed during inference: {context[path]}")


def require_valid_metrics(metrics):
    """Reject incomplete or non-finite metrics before they are published as complete."""
    if not metrics.get("validation_complete") or metrics["episodes_completed"] < metrics["episodes_requested"]:
        raise ValueError("Incomplete inference metrics cannot be published as complete")
    measured = metrics["inference_deterministic"]
    for key in ("E_xy_mm", "E_z_mm", "norm_mm"):
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in measured[key].values()):
            raise ValueError(f"Invalid primary inference metric: {key}")
    for key in ("success_rate", "reach_rate", "near_fail_rate", "stay20_rate"):
        if not math.isfinite(float(measured[key])) or not 0 <= float(measured[key]) <= 1:
            raise ValueError(f"Invalid inference rate: {key}")


def create_evaluation_directory(checkpoint, output=None):
    """One attempt per directory; periodic and final checkpoints have distinct stems."""
    from datetime import datetime
    from rl_env.project_config import EVAL_ROOT
    model = Path(checkpoint)
    if output is None:
        parent = Path(EVAL_ROOT) / model.parent.name / model.stem
        output = parent / ("inference_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing evaluation artifacts: {output}; select a new --out.")
    output.mkdir(parents=True, exist_ok=True)
    return str(output)
