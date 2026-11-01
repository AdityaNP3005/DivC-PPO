from __future__ import annotations

"""Reset-time object placement: the target object (and optional bracket) directly under
the gripper projection, with the configured XY/Z or fixed-radius randomization."""

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def place_cylinder_under_gripper(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    bracket_cfg: SceneEntityCfg | None = None,
    floor_z_m: float = 0.0,
    object_height_m: float = 0.10,
    clearance_m: float = 0.002,
    cylinder_floor_z_m: float = 0.0,
    x_offset_range_m: tuple[float, float] = (0.0, 0.0),
    y_offset_range_m: tuple[float, float] = (0.0, 0.0),
    z_offset_range_m: tuple[float, float] = (0.0, 0.0),
    ee_local_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radial_offset_m: float | None = None,
) -> None:
    """Place the cylinder on the floor below the gripper projection, with the sampled offset.

    - Bracket: XY = gripper XY, Z = floor + clearance
    - Cylinder: XY = gripper XY, Z = cylinder_floor_z_m + 0.5*height + clearance
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    cylinder: RigidObject = env.scene[cylinder_cfg.name]
    bracket: RigidObject | None = env.scene[bracket_cfg.name] if bracket_cfg is not None else None

    ee_pos_w = robot.data.body_pos_w[env_ids, robot_cfg.body_ids[0]]  # type: ignore[index]
    ee_quat_w = robot.data.body_quat_w[env_ids, robot_cfg.body_ids[0]]  # type: ignore[index]
    target_pos_w = ee_pos_w.clone()
    if ee_local_offset_m != (0.0, 0.0, 0.0):
        offset_b = torch.tensor(ee_local_offset_m, device=env.device, dtype=target_pos_w.dtype).unsqueeze(0)
        offset_b = offset_b.repeat(target_pos_w.shape[0], 1)
        rot_mat = matrix_from_quat(ee_quat_w)
        offset_w = torch.bmm(rot_mat, offset_b.unsqueeze(-1)).squeeze(-1)
        target_pos_w = target_pos_w + offset_w
    if radial_offset_m is not None and float(radial_offset_m) > 0.0:
        # PLAY stress test: place the object at the FARTHEST trained distance (fixed radius),
        # but a RANDOM direction each reset — exercises the policy at the edge of its trained
        # range from every angle. Per-axis ranges above are ignored in this mode.
        ang = torch.empty((len(env_ids),), device=env.device, dtype=target_pos_w.dtype).uniform_(
            0.0, 2.0 * math.pi
        )
        target_pos_w[:, 0].add_(float(radial_offset_m) * torch.cos(ang))
        target_pos_w[:, 1].add_(float(radial_offset_m) * torch.sin(ang))
    else:
        if x_offset_range_m != (0.0, 0.0):
            target_pos_w[:, 0].add_(
                torch.empty((len(env_ids),), device=env.device, dtype=target_pos_w.dtype).uniform_(
                    float(x_offset_range_m[0]), float(x_offset_range_m[1])
                )
            )
        if y_offset_range_m != (0.0, 0.0):
            target_pos_w[:, 1].add_(
                torch.empty((len(env_ids),), device=env.device, dtype=target_pos_w.dtype).uniform_(
                    float(y_offset_range_m[0]), float(y_offset_range_m[1])
                )
            )
    z_offset = torch.zeros((len(env_ids),), device=env.device, dtype=target_pos_w.dtype)
    if z_offset_range_m != (0.0, 0.0):
        z_offset = torch.empty((len(env_ids),), device=env.device, dtype=target_pos_w.dtype).uniform_(
            float(z_offset_range_m[0]), float(z_offset_range_m[1])
        )
    target_pos_w[:, 2] = float(cylinder_floor_z_m) + 0.5 * float(object_height_m) + float(clearance_m) + z_offset

    # Place cylinder
    default_state_cyl = cylinder.data.default_root_state[env_ids].clone()
    quat_w_cyl = default_state_cyl[:, 3:7]
    pose_cyl = torch.cat([target_pos_w, quat_w_cyl], dim=-1)
    cylinder.write_root_pose_to_sim(pose_cyl, env_ids=env_ids)
    vel_w = torch.zeros((len(env_ids), 6), device=env.device, dtype=default_state_cyl.dtype)
    cylinder.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    # Place bracket at floor level (same XY, Z on floor)
    if bracket is not None:
        bracket_pos_w = target_pos_w.clone()
        bracket_pos_w[:, 2] = float(floor_z_m) + float(clearance_m)
        default_state_br = bracket.data.default_root_state[env_ids].clone()
        quat_w_br = default_state_br[:, 3:7]
        pose_br = torch.cat([bracket_pos_w, quat_w_br], dim=-1)
        bracket.write_root_pose_to_sim(pose_br, env_ids=env_ids)
        bracket.write_root_velocity_to_sim(vel_w.clone(), env_ids=env_ids)
