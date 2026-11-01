from __future__ import annotations

"""EE→target error sources for the reward terms, plus GT target geometry and the
critic-only GT observation terms.

Reward terms call `_reward_error(..., error_source)`: "gt" is the privileged world-frame
error (asymmetric arms), "vision" the metric-equivalent YOLO error (pure-vision arms).
RewardsCfg sets error_source from config.reward_units.reward_error_source.
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from config import (
    BBOX_TARGET_CENTER_PX,
    BBOX_TARGET_DEPTH_M,
    VISION_REWARD_NO_DET_ERROR_M,
    VISION_REWARD_XY_GAIN_M,
    YOLO_CACHE_NORM_DEPTH_M,
    YOLO_CACHE_NORM_PX,
)
from rl_env.perception.yolo_cache import _get_yolo_detection_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_VISION_REWARD_SENSOR_CFG = SceneEntityCfg("wrist_camera")


def _gt_object_top_z_w(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    object_height_m: float = 0.10,
) -> torch.Tensor:
    """Cylinder top Z in world frame. CylinderCfg root pose is at the center."""
    return env.scene[object_cfg.name].data.root_pos_w[:, 2] + 0.5 * float(object_height_m)


def _gt_target_pos_w(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
) -> torch.Tensor:
    """GT hover target: cylinder XY, cylinder top + clearance in Z."""
    target = env.scene[object_cfg.name].data.root_pos_w[:, :3].clone()
    target[:, 2] = _gt_object_top_z_w(env, object_cfg, object_height_m) + float(clearance_above_m)
    return target


def gt_task_target_pos_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """GT hover target [N,3] of the task cylinder with the configured geometry."""
    from config.task import GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M, OBJECT_CYL_HEIGHT_M

    return _gt_target_pos_w(env, SceneEntityCfg("cylinder"), OBJECT_CYL_HEIGHT_M, GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M)


def gt_target_error_w(env: "ManagerBasedRLEnv", ee_body_id: int) -> torch.Tensor:
    """GT target − EE [N,3], world frame (monitor/validator/play sign).

    Reward terms use the opposite sign (_gt_ee_target_error_w = EE − target).
    """
    return gt_task_target_pos_w(env) - env.scene["robot"].data.body_pos_w[:, ee_body_id]


def _gt_ee_target_error_w(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
) -> torch.Tensor:
    """GT error ee_xyz - target_xyz in world frame, metres."""
    ee_pos_w = env.scene[ee_cfg.name].data.body_pos_w[:, ee_cfg.body_ids[0]]  # type: ignore[index]
    return ee_pos_w - _gt_target_pos_w(env, object_cfg, object_height_m, clearance_above_m)


def _vision_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """YOLO error vs the visual target as [dx, dy, ddepth] in metre-equivalent units.

    XY is the normalized-pixel center error × gain (≈ metres near the tolerance); Z is
    the depth error in metres. Reward terms use |error|, so the image→world axis sign is
    irrelevant. Lost detection is treated as "far" on every axis to push re-acquisition.
    """
    cache = _get_yolo_detection_cache(env, _VISION_REWARD_SENSOR_CFG)
    centers = cache["center_norm"]
    depth = cache["depth_area_norm"]

    target_cx = float(BBOX_TARGET_CENTER_PX[0]) / YOLO_CACHE_NORM_PX
    target_cy = float(BBOX_TARGET_CENTER_PX[1]) / YOLO_CACHE_NORM_PX
    target_depth = float(BBOX_TARGET_DEPTH_M) / YOLO_CACHE_NORM_DEPTH_M

    error = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    error[:, 0] = (centers[:, 0] - target_cx) * float(VISION_REWARD_XY_GAIN_M)
    error[:, 1] = (centers[:, 1] - target_cy) * float(VISION_REWARD_XY_GAIN_M)
    error[:, 2] = (depth[:, 0] - target_depth) * YOLO_CACHE_NORM_DEPTH_M
    error[cache["has_det"] < 0.5] = float(VISION_REWARD_NO_DET_ERROR_M)
    return error


def _reward_error(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_height_m: float,
    clearance_above_m: float,
    error_source: str,
) -> torch.Tensor:
    """EE→target error [N,3] from the configured source ("gt" or "vision")."""
    if error_source == "vision":
        return _vision_error(env)
    if error_source == "gt":
        return _gt_ee_target_error_w(env, ee_cfg, object_cfg, object_height_m, clearance_above_m)
    raise ValueError(f"error_source must be 'gt' or 'vision', got {error_source!r}")


def gt_ee_target_error_obs(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    clip: float = 10.0,
) -> torch.Tensor:
    """Privileged critic obs: signed GT error target_xyz - ee_xyz, in meters.

    Actor networks mask this tail out. The critic can use it to estimate value
    from clean geometry while the actor still relies on deployable visual obs.
    """
    target_minus_ee = -_gt_ee_target_error_w(env, ee_cfg, object_cfg, object_height_m, clearance_above_m)
    target_minus_ee = torch.nan_to_num(target_minus_ee, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(target_minus_ee, -float(clip), float(clip))


def gt_target_pos_obs(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    clip: float = 10.0,
) -> torch.Tensor:
    """GT target position in world frame: cylinder XY, cylinder top + clearance Z."""
    target = _gt_target_pos_w(env, object_cfg, object_height_m, clearance_above_m)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(target, -float(clip), float(clip))
