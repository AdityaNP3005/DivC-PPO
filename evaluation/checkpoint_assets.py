"""Object/detector bootstrap and matching normalization, without Isaac/config imports."""
import glob
import os
from pathlib import Path
import re
import sys

import yaml
from evaluation.agent_metadata import parse_agent_yaml


def checkpoint_from_argv(default=None):
    selected = default
    for index, value in enumerate(sys.argv):
        if value == "--checkpoint" and index + 1 < len(sys.argv):
            selected = sys.argv[index + 1]
        if value.startswith("--checkpoint="):
            selected = value.split("=", 1)[1]
    return selected


def bootstrap_inference_assets(default=None):
    """Select CLI/configured/latest once, before perception/task defaults freeze.

    The selected absolute path becomes PLAY_DEFAULTS['checkpoint']; consumers
    must reuse it rather than searching for latest after config import.
    """
    if Path(sys.argv[0]).name != "evaluate.py":
        return default
    selected = checkpoint_from_argv(default) or default
    if not selected:
        from rl_env.project_config import log_base
        candidates = sorted(glob.glob(os.path.join(log_base(), "*/model.zip")))
        selected = candidates[-1] if candidates else None
    if selected:
        selected = os.path.abspath(selected)
        restore_object_env(selected)
    return selected


def restore_object_env(checkpoint):
    """Nonempty process overrides win; blank forwarded values restore saved metadata.

    A complete object bundle explicitly selects its object, including the
    primitive cylinder's empty USD. An empty wrapper-only bundle is not an override.
    """
    if not checkpoint:
        return
    path = Path(checkpoint).parent / "params" / "agent.yaml"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    data = parse_agent_yaml(text, legacy_errors=(yaml.constructor.ConstructorError,))
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint metadata must be a mapping: {path}")
    if not os.environ.get("REWARD_ERROR_SOURCE", "").strip():
        os.environ["REWARD_ERROR_SOURCE"] = data.get(
            "reward_error_source", "gt" if data.get("asymmetric_critic", False) else "vision"
        )
    from rl_env.project_config import local_asset_path
    object_bundle = "OBJECT_USD" in os.environ and all(
        os.environ.get(key, "").strip()
        for key in ("OBJECT_USD_SCALE", "OBJECT_CYL_HEIGHT_M", "CYLINDER_FLOOR_Z_M", "YOLO_MODEL_PATH")
    )
    for key, saved in (
        ("OBJECT_USD", "object_usd"),
        ("OBJECT_USD_SCALE", "object_usd_scale"),
        ("OBJECT_CYL_HEIGHT_M", "object_cyl_height_m"),
        ("CYLINDER_FLOOR_Z_M", "cylinder_floor_z_m"),
        ("YOLO_MODEL_PATH", "yolo_model_path"),
    ):
        value = os.environ.get(key, "").strip()
        if not value and not (key == "OBJECT_USD" and object_bundle):
            value = data.get(saved)
        if value is not None:
            if key in ("OBJECT_USD", "YOLO_MODEL_PATH"):
                value = local_asset_path(str(value))
            os.environ[key] = str(value)


def bootstrap_training_assets(default=None):
    """Called before perception/task config freezes CLI or configured resume assets."""
    if Path(sys.argv[0]).name == "train_ur10e.py":
        restore_object_env(checkpoint_from_argv(default))


def require_detector(path):
    """Missing explicit/saved detectors never fall back to another object model."""
    from rl_env.project_config import local_asset_path
    resolved = local_asset_path(str(path).strip())
    if not Path(resolved).is_file():
        raise FileNotFoundError(f"YOLO detector not found: {resolved}; no detector fallback is allowed.")
    return resolved


def resolve_vecnormalize_path(checkpoint):
    """Use the matching pair for periodic models; final/legacy models may use root stats."""
    path = Path(checkpoint)
    stem = path.stem if path.suffix == ".zip" else path.name
    candidates = [path.parent / f"{stem}_vecnormalize.pkl"]
    periodic = re.fullmatch(r"(.+)_(\d+)_steps(?:zip)?", stem)
    if periodic:
        prefix, steps = periodic.groups()
        candidates.extend((
            path.parent / f"{prefix}_{steps}_stepsvecnormalize_pkl",  # old project suffix bug
            path.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl",  # stock SB3 naming
        ))
    else:
        candidates.append(path.parent / "vecnormalize.pkl")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Matching VecNormalize not found for {checkpoint}. Checked: "
        + ", ".join(map(str, candidates))
        + (". Periodic checkpoints cannot use final model statistics." if periodic else "")
    )
