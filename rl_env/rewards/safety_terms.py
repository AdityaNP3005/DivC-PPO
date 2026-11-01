from __future__ import annotations

"""Safety reward terms: per-axis action rate, YOLO no-detection penalty and joint-deviation clamp penalty."""

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from rl_env.perception.yolo_cache import _get_yolo_detection_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_axis(env: "ManagerBasedRLEnv", axis: int) -> torch.Tensor:
    """Per-axis form of IsaacLab's action_rate_l2; summing XYZ is identical."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.square(delta[:, int(axis)])


def yolo_no_detection_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    penalty_value: float = 1.0,
    conf_threshold: float = 0.35,
) -> torch.Tensor:
    """Return a positive mask when detection is absent or confidence is below threshold.

    The reward term weight is negative, so invalid detection becomes a penalty.

    - Detection with conf >= threshold -> 0.0
    - No detection or conf < threshold -> penalty_value (default 1.0)
    Shape: [num_envs]
    """
    cache = _get_yolo_detection_cache(env, sensor_cfg)
    conf = cache["conf"]
    valid_detection = (cache["has_det"] > 0.5) & (conf >= float(conf_threshold))
    return (~valid_detection).to(conf.dtype) * float(penalty_value)


def joint_limit_clamp_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_deviation: float = 2.5,
    action_term_name: str = "arm_action",
    action_joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalty per joint that deviates more than max_deviation rad from its default position.

    MENGAPA BUKAN soft_joint_pos_limits:
        The UR10e soft_joint_pos_limits are ±2π (±6.28 rad); the arm can fold long before
        reaching them, so a penalty on the soft limits would never trigger.

    Instead, compare q_target with q_default ± max_deviation.
        q_default = joint position at episode reset (init_state in rl_env/env/scene_cfg.py)
        max_deviation = largest allowed deviation

    Pilih max_deviation:
        - Reset randomization: ±0.3 rad from default
        - Normal motion to the target: ~1.0–1.5 rad from default
        - Fold/collapse: usually >2.0 rad from default
        -> max_deviation=2.5 rad allows reaching and prevents folding

    Formula per env:
        q_target  = q_current + processed_actions
        hit_upper = q_target > q_default + max_deviation  (bool per joint)
        hit_lower = q_target < q_default - max_deviation  (bool per joint)
        penalty   = (hit_upper | hit_lower).float().sum(dim=-1)

    Output = number of joints beyond the limit.
    With weight=-20: one joint over the limit gives -20 per step.
    """
    from isaaclab.assets import Articulation
    asset: Articulation = env.scene[asset_cfg.name]

    q_current = asset.data.joint_pos[:, asset_cfg.joint_ids]       # [N, num_joints]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # [N, num_joints]

    # Joint-mode uses processed_actions as delta joint. Cartesian-mode exposes
    # joint_delta_actions because processed_actions is xyz delta.
    action_term = env.action_manager.get_term(action_term_name)
    processed = getattr(action_term, "joint_delta_actions", None)
    if processed is None:
        processed = getattr(action_term, "processed_actions", None)
    if processed is None:
        return torch.zeros(env.num_envs, device=env.device)

    # If the action joints differ from the clamped subset, map them by joint name.
    if processed.shape[1] != q_current.shape[1]:
        processed_subset = torch.zeros_like(q_current)
        asset_joint_names = list(asset.data.joint_names)
        action_joint_name_list = list(action_joint_names or [])
        action_name_to_idx = {name: idx for idx, name in enumerate(action_joint_name_list)}
        for subset_idx, joint_id in enumerate(asset_cfg.joint_ids):
            joint_name = asset_joint_names[joint_id]
            action_idx = action_name_to_idx.get(joint_name)
            if action_idx is not None and action_idx < processed.shape[1]:
                processed_subset[:, subset_idx] = processed[:, action_idx]
        processed = processed_subset

    q_target = q_current + processed

    # Joints beyond max_deviation from their default position
    hit = (q_target > q_default + max_deviation) | (q_target < q_default - max_deviation)

    return hit.float().sum(dim=-1)  # number of joints over the limit per env
