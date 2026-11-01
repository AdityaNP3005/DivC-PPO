from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import SceneEntityCfg
from rl_env.env_ids import env_ids_tensor as _env_ids_tensor


def _joint_position_targets(env, env_ids, asset_cfg, joint_targets):
    """(asset, env ids, joint ids, [n_env, n_joint] targets), or None when nothing to set.

    Build data only; reset and hold keep their distinct simulator writes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _env_ids_tensor(env, env_ids)
    joint_names = list(asset_cfg.joint_names) if asset_cfg.joint_names is not None else []
    if ids.numel() == 0 or not joint_names:
        return None
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    target = torch.zeros((ids.numel(), len(joint_names)), device=env.device, dtype=asset.data.joint_pos.dtype)
    for col, name in enumerate(joint_names):
        target[:, col] = float(joint_targets.get(name, 0.0))
    return asset, ids, joint_ids, target


def reset_joint_targets_to_positions(
    env,
    env_ids: torch.Tensor | slice | Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    joint_targets: dict[str, float],
    zero_velocity: bool = True,
) -> None:
    """Reset selected joints and actuator targets to explicit positions."""
    built = _joint_position_targets(env, env_ids, asset_cfg, joint_targets)
    if built is None:
        return
    asset, ids, joint_ids, target = built

    zero_vel = torch.zeros_like(target)
    asset.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids, env_ids=ids)
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=ids)
    if zero_velocity:
        asset.set_joint_velocity_target(zero_vel, joint_ids=joint_ids, env_ids=ids)
    asset.write_data_to_sim()


def hold_joint_setpoints(
    env,
    env_ids: torch.Tensor | slice | Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    joint_targets: dict[str, float],
    zero_velocity: bool = True,
) -> None:
    """Hold gripper joints at fixed setpoints while Cartesian IK owns the arm."""
    built = _joint_position_targets(env, env_ids, asset_cfg, joint_targets)
    if built is None:
        return
    asset, ids, joint_ids, target = built

    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=ids)
    if zero_velocity:
        asset.set_joint_velocity_target(torch.zeros_like(target), joint_ids=joint_ids, env_ids=ids)
    asset.write_data_to_sim()
