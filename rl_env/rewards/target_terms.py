from __future__ import annotations

"""EE→target reward terms: progress, bounded abs, exponential, idle, success-once,
stay, alive, plus the GT-only low-clearance safety term.

Every target term reads its error through `_reward_error(..., error_source)`, so the
same shaping applies to the GT (asymmetric) and vision (pure-vision) arms.
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from rl_env.env_ids import just_reset
from rl_env.rewards.error_sources import _gt_object_top_z_w, _reward_error
from rl_env.success_metrics import per_axis_in_band

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _axis_ids(active_axes: str) -> list[int]:
    """Indices of the active axes, always in x→y→z order."""
    axes = str(active_axes).lower()
    return [idx for idx, name in enumerate("xyz") if name in axes]


def _normalized_axis_error(
    error: torch.Tensor,
    active_axes: str,
    xy_scale_m: float,
    z_scale_m: float,
    z_asymmetric: bool = False,
) -> torch.Tensor:
    """Sum over active axes of |error| / scale. z_asymmetric counts only EE-above-target Z."""
    scales = (max(float(xy_scale_m), 1e-6),) * 2 + (max(float(z_scale_m), 1e-6),)
    dist = torch.zeros_like(error[:, 0])
    for idx in _axis_ids(active_axes):
        axis_error = torch.clamp(error[:, idx], min=0.0) if idx == 2 and z_asymmetric else torch.abs(error[:, idx])
        dist = dist + axis_error / scales[idx]
    return dist


def target_progress_reward(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    active_axes: str = "xyz",
    xy_scale_m: float = 0.30,
    z_scale_m: float = 0.30,
    z_asymmetric: bool = False,
) -> torch.Tensor:
    """Best-so-far progress: + when beating the episode's best distance, − when falling behind it."""
    error = _reward_error(env, ee_cfg, object_cfg, object_height_m, clearance_above_m, error_source)
    dist = _normalized_axis_error(error, active_axes, xy_scale_m, z_scale_m, z_asymmetric)

    key = (
        str(active_axes).lower(),
        float(object_height_m),
        float(clearance_above_m),
        float(xy_scale_m),
        float(z_scale_m),
        bool(z_asymmetric),
    )
    if not hasattr(env, "_best_gt_target_error_by_term"):
        env._best_gt_target_error_by_term = {}
    best_by_term = env._best_gt_target_error_by_term
    best = best_by_term.get(key)
    if best is None or best.shape != dist.shape:
        best = dist.detach().clone()

    reward = best - dist
    reset_now = just_reset(env)
    reward[reset_now] = 0.0
    best = torch.minimum(best, dist.detach())
    best[reset_now] = dist[reset_now].detach()
    best_by_term[key] = best
    return reward


def target_abs_penalty(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    active_axes: str = "xyz",
    xy_scale_m: float = 0.30,
    z_scale_m: float = 0.30,
    z_asymmetric: bool = False,
) -> torch.Tensor:
    """Normalized distance clamped to [0, 1]; the negative weight makes it a penalty."""
    error = _reward_error(env, ee_cfg, object_cfg, object_height_m, clearance_above_m, error_source)
    return torch.clamp(_normalized_axis_error(error, active_axes, xy_scale_m, z_scale_m, z_asymmetric), 0.0, 1.0)


def target_exp_reward(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    active_axes: str = "xyz",
    xy_scale_m: float = 0.05,
    z_scale_m: float = 0.05,
) -> torch.Tensor:
    """Mean over active axes of exp(−|error| / scale): steep near zero, flat far away."""
    ids = _axis_ids(active_axes)
    error = _reward_error(env, ee_cfg, object_cfg, object_height_m, clearance_above_m, error_source).abs()[:, ids]
    axis_scales = (float(xy_scale_m), float(xy_scale_m), float(z_scale_m))
    scales = torch.tensor([axis_scales[idx] for idx in ids], device=env.device, dtype=torch.float32).unsqueeze(0)
    return torch.exp(-error / torch.clamp(scales, min=1e-6)).mean(dim=1)


def target_idle_penalty(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    idle_xy_threshold_m: float = 0.01,
    idle_z_threshold_m: float = 0.01,
    idle_action_threshold: float = 0.003,
    action_term_name: str = "arm_action",
    active_axes: str = "xyz",
) -> torch.Tensor:
    """Count of active axes that are still off target while their action is ~zero."""
    error = _reward_error(env, ee_cfg, object_cfg, object_height_m, clearance_above_m, error_source).abs()
    action_term = env.action_manager.get_term(str(action_term_name))
    action = getattr(action_term, "processed_actions", None)
    if action is None:
        action = getattr(action_term, "joint_delta_actions", None)
    if action is None or action.shape[-1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    thresholds = (float(idle_xy_threshold_m), float(idle_xy_threshold_m), float(idle_z_threshold_m))
    penalty = torch.zeros(env.num_envs, device=env.device)
    for idx in _axis_ids(active_axes):
        if idx < action.shape[-1]:
            penalty += ((error[:, idx] > thresholds[idx]) & (action[:, idx].abs() < float(idle_action_threshold))).float()
    return penalty


def _at_target(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float,
    clearance_above_m: float,
    xy_tol_m: float,
    z_tol_m: float,
) -> torch.Tensor:
    error = _reward_error(env, ee_cfg, object_cfg, object_height_m, clearance_above_m, error_source)
    return per_axis_in_band(error, xy_tol_m, z_tol_m)


def target_success_once_bonus(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    xy_tol_m: float = 0.008,
    z_tol_m: float = 0.008,
) -> torch.Tensor:
    """1 on the first step of each episode that enters the target band."""
    at_target = _at_target(env, ee_cfg, object_cfg, error_source, object_height_m, clearance_above_m, xy_tol_m, z_tol_m)
    if not hasattr(env, "_gt_target_success_given"):
        env._gt_target_success_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    env._gt_target_success_given[just_reset(env)] = False
    first_success = at_target & ~env._gt_target_success_given
    env._gt_target_success_given |= at_target
    return first_success.float()


def target_stay_reward(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    xy_tol_m: float = 0.008,
    z_tol_m: float = 0.008,
) -> torch.Tensor:
    """step_dt while inside the target band.

    RewardManager multiplies by step_dt again, so the effective rate is weight·dt² (see
    docs/REWARD_DESIGN.md); kept as-is to match the paper checkpoints.
    """
    at_target = _at_target(env, ee_cfg, object_cfg, error_source, object_height_m, clearance_above_m, xy_tol_m, z_tol_m)
    return at_target.float() * float(env.step_dt)


def target_alive_penalty(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    error_source: str,
    object_height_m: float = 0.10,
    clearance_above_m: float = 0.12,
    xy_tol_m: float = 0.008,
    z_tol_m: float = 0.008,
) -> torch.Tensor:
    """1 while outside the target band; the negative weight makes it time pressure."""
    at_target = _at_target(env, ee_cfg, object_cfg, error_source, object_height_m, clearance_above_m, xy_tol_m, z_tol_m)
    return (~at_target).float()


def gt_low_clearance_penalty(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_height_m: float = 0.10,
    min_clearance_above_top_m: float = 0.002,
) -> torch.Tensor:
    """1 when the EE is below cylinder top + minimum clearance (always GT; GT arms only)."""
    ee_z = env.scene[ee_cfg.name].data.body_pos_w[:, ee_cfg.body_ids[0], 2]  # type: ignore[index]
    min_z = _gt_object_top_z_w(env, object_cfg, object_height_m) + float(min_clearance_above_top_m)
    return (ee_z < min_z).float()
