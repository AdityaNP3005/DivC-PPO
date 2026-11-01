from __future__ import annotations

"""Tunable configuration for single-actor PPO and DivC-PPO.

This package holds the knobs a researcher tweaks per experiment, split by concern:
    training      — TRAIN_DEFAULTS, PPO_DEFAULTS, RESUME_DEFAULTS, PLAY_DEFAULTS (camera/YOLO defaults for validation)
    policy        — ASYMMETRIC_CRITIC_DEFAULTS, CREDIT_DEFAULTS
    action        — CARTESIAN_ACTION_DEFAULTS (position scale, IK damping, joint speed)
    perception    — camera, YOLO defaults + publish_yolo_env, cache normalization, visual target
    env_flags     — env-var parsing helpers + resolve_yolo_device

Fixed-library code (IK math, PPO loop, networks, YOLO runtime) must import its
numbers from here and never hardcode them. Runtime callers import config directly.
"""

from .training import (
    TRAIN_DEFAULTS,
    PPO_DEFAULTS,
    RESUME_DEFAULTS,
    PLAY_DEFAULTS,
)
from .policy import (
    ASYMMETRIC_CRITIC_DEFAULTS,
    CREDIT_DEFAULTS,
)
from .action import (
    CARTESIAN_ACTION_DEFAULTS,
)
from .perception import (
    CAMERA_PRIM_PATH,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_DEFAULTS,
    YOLO_DEFAULTS,
    publish_yolo_env,
    YOLO_CACHE_NORM_PX,
    YOLO_CACHE_NORM_DEPTH_M,
    YOLO_TRAIN_BBOX_CENTER_JITTER_STD,
    YOLO_DEPTH_PERCENTILE,
    BBOX_TARGET_CENTER_PX,
    BBOX_TARGET_DEPTH_M,
    YOLO_ROI_ENABLED,
    YOLO_GRIPPER_EXCLUSION_MODE,
    YOLO_GRIPPER_EXCLUSION_MASK_VERSION,
    YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX,
    YOLO_GRIPPER_EXCLUSION_POLYGONS_PX,
    VISION_REWARD_XY_GAIN_M,
    VISION_REWARD_NO_DET_ERROR_M,
)
from .env_flags import (
    env_bool,
    env_str,
    env_float,
    env_int,
    env_scale3,
    resolve_yolo_device,
)
from . import rewards_weights  # noqa: F401  (accessed as config.rewards_weights.*)
from . import task  # noqa: F401  (task constants accessed as config.task.* or imported direct)

__all__ = [
    "TRAIN_DEFAULTS",
    "PPO_DEFAULTS",
    "RESUME_DEFAULTS",
    "PLAY_DEFAULTS",
    "ASYMMETRIC_CRITIC_DEFAULTS",
    "CREDIT_DEFAULTS",
    "CARTESIAN_ACTION_DEFAULTS",
    "CAMERA_PRIM_PATH",
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "CAMERA_DEFAULTS",
    "YOLO_DEFAULTS",
    "publish_yolo_env",
    "YOLO_CACHE_NORM_PX",
    "YOLO_CACHE_NORM_DEPTH_M",
    "YOLO_TRAIN_BBOX_CENTER_JITTER_STD",
    "YOLO_DEPTH_PERCENTILE",
    "BBOX_TARGET_CENTER_PX",
    "BBOX_TARGET_DEPTH_M",
    "YOLO_ROI_ENABLED",
    "YOLO_GRIPPER_EXCLUSION_MODE",
    "YOLO_GRIPPER_EXCLUSION_MASK_VERSION",
    "YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX",
    "YOLO_GRIPPER_EXCLUSION_POLYGONS_PX",
    "VISION_REWARD_XY_GAIN_M",
    "VISION_REWARD_NO_DET_ERROR_M",
    "env_bool",
    "env_str",
    "env_float",
    "env_int",
    "env_scale3",
    "resolve_yolo_device",
    "rewards_weights",
    "task",
]
