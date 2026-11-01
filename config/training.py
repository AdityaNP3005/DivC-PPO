from __future__ import annotations

"""Tunable training/runtime knobs: train, PPO optimizer, resume, and play defaults.

TUNABLE — edit values here for the values you normally use. The runtime CLI is
intentionally thin; use this file for normal train/play changes.
"""

from .env_flags import env_float, env_int
from .reward_units import validate_reward_units
import os

# PPO-only project. No alternate algorithm is wired into this repo.

# Main train defaults. Set `checkpoint` to a model.zip path when you want resume
# to be the default behavior; leave it as None for fresh training.
TRAIN_DEFAULTS = {
    "num_envs": env_int("NUM_ENVS", 128),
    # Env-overridable (MAX_STEPS) for explicit sweeps/smoke tests. Unset → the tuned default below.
    "max_steps": env_int("MAX_STEPS", 1_000_000),
    "checkpoint_interval": None,
    "keep_checkpoints": 5,
    "checkpoint": None,
    "seed": 42,
    "PPO_head": "single",  # PPO, PPO-asym and DivC-PPO use one actor
    # The paper uses the symmetric critic (actor and critic see the same 12-D observation).
    # env ASYMMETRIC_CRITIC=1 enables the privileged critic variant, which the paper does not use.
    "asymmetric_critic": False,
    # Env-overridable (GPU) so a campaign can be pinned to another free card without
    # touching this file; every launcher defaults to GPU 1.
    "gpu": env_int("GPU", 1),
    "camera_timeout_sec": 8.0,
    "scalar_reward_units": validate_reward_units(
        (os.environ.get("PPO_REWARD_UNITS") or "legacy_rate").strip()
    ),
}

# Checkpoint selection must precede perception/task imports. Edit the play
# checkpoint here; remaining camera/detector defaults are populated below.
PLAY_DEFAULTS = {
    "checkpoint": None,
}

# Restore the selected training/replay assets before config values freeze.
from evaluation.checkpoint_assets import bootstrap_training_assets, bootstrap_inference_assets
bootstrap_training_assets(TRAIN_DEFAULTS["checkpoint"])
PLAY_DEFAULTS["checkpoint"] = bootstrap_inference_assets(PLAY_DEFAULTS["checkpoint"])
from .perception import CAMERA_DEFAULTS, YOLO_DEFAULTS

# PPO optimizer/sampling defaults. Use None for auto-scaling based on num_envs.
PPO_DEFAULTS = {
    "n_steps": env_int("PPO_N_STEPS", 0) or None,
    "batch_size": env_int("PPO_BATCH_SIZE", 0) or None,
    "n_epochs": 5,
    "learning_rate": 2e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": None,
    # Env-overridable (PPO_ENT_COEF / PPO_LOG_STD_INIT) for explicit sweeps.
    # Unset → the tuned defaults below, which the paper runs use directly.
    "ent_coef": env_float("PPO_ENT_COEF", 0.005),
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    # Separate actor/critic MLPs so critic (9D privileged) and actor (12D visual)
    # have independent representations. pi=actor, vf=critic.
    "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    "log_std_init": env_float("PPO_LOG_STD_INIT", -1.0),
}

# Optional resume-only overrides on top of PPO_DEFAULTS. Leave None/False unless
# resume should differ from fresh training settings.
RESUME_DEFAULTS = {
    "learning_rate": None,
    "clip_range": None,
    "ent_coef": None,
    "vf_coef": None,
    "n_epochs": None,
    "batch_size": None,
    "target_kl": None,
    "log_std_init": None,
}

PLAY_DEFAULTS.update({
    "num_episodes": 5,
    "video_length": 220,
    "seed": TRAIN_DEFAULTS["seed"],
    "gpu": TRAIN_DEFAULTS["gpu"],
    "camera_resolution": CAMERA_DEFAULTS["resolution"],
    "camera_update_period": CAMERA_DEFAULTS["update_period"],
    "yolo_model_path": YOLO_DEFAULTS["model_path"],
    "yolo_conf": YOLO_DEFAULTS["conf"],
    "yolo_iou": YOLO_DEFAULTS["iou"],
    "yolo_device": YOLO_DEFAULTS["device"],
    "yolo_max_det": YOLO_DEFAULTS["max_det"],
    "no_video": False,
    "no_env_video": False,
    "no_camera_det": False,
    "real_time": False,
})

VALIDATION_DEFAULTS = {
    "num_envs": env_int("EVAL_NUM_ENVS", 128),
    "min_episodes": env_int("EVAL_MIN_EPISODES", 1000),
    "max_steps": env_int("EVAL_MAX_STEPS", 0),
}
