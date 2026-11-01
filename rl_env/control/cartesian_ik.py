from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat


def jacobian_body_index(asset, body_idx: int) -> int:
    """PhysX Jacobian row block for a body; a fixed base has no root block."""
    return body_idx - 1 if asset.is_fixed_base else body_idx


def joint_position_limits(asset_data):
    """[N, joints, 2] position limits: soft limits when present, else hard limits (or None)."""
    limits = getattr(asset_data, "soft_joint_pos_limits", None)
    if limits is None:
        limits = getattr(asset_data, "joint_pos_limits", None)
    return limits


def dls_delta_q(jacobian: torch.Tensor, task: torch.Tensor, damping: float) -> torch.Tensor:
    """Damped least squares: dq = J^T (J J^T + damping^2 I)^-1 task, for [N,6,n] J and [N,6] task."""
    j_t = jacobian.transpose(1, 2)
    eye6 = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0).repeat(jacobian.shape[0], 1, 1)
    lhs = torch.bmm(jacobian, j_t) + (damping * damping) * eye6
    return torch.bmm(j_t, torch.linalg.solve(lhs, task.unsqueeze(-1))).squeeze(-1)


class CartesianDeltaAction(ActionTerm):
    """Policy-controlled XYZ deltas with bounded six-joint orientation-hold IK."""

    _asset: Articulation

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if not len(body_ids) == 1:
            raise ValueError(
                f"Expected exactly one body for CartesianDeltaAction body_name={self.cfg.body_name}. "
                f"Found {len(body_ids)}: {body_names}"
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = jacobian_body_index(self._asset, self._body_idx)
        if self._asset.is_fixed_base:
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_joint_pos = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_joint_delta = torch.zeros_like(self._target_joint_pos)

        self._position_scale = torch.tensor(self.cfg.position_scale_m, device=self.device, dtype=torch.float32).view(1, 3)
        control_dt = float(getattr(env, "step_dt", env.sim.get_physics_dt()))
        self._max_joint_step_rad = math.radians(float(self.cfg.max_joint_speed_deg_s)) * control_dt

        self._z_target_w = float(self.cfg.z_target_w)
        # ── Orientation-hold IK: full EE rotation captured at episode reset (per-env) ──
        self._target_rot_w: torch.Tensor | None = None
        self._ori_captured: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_delta_actions(self) -> torch.Tensor:
        return self._last_joint_delta

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        safe_actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        delta_pos_w = safe_actions * self._position_scale

        # ── Policy-Z floor clamp ──
        # Don't let the policy command the EE below the Z setpoint (z_target_w).
        # Without this the policy overshoots past the target and bounces. The EE
        # may descend down to the setpoint but never below it.
        ee_z_w = self._asset.data.body_pos_w[:, self._body_idx, 2]
        min_delta_z = float(self._z_target_w) - ee_z_w  # delta that lands exactly at the floor
        delta_pos_w[:, 2] = torch.maximum(delta_pos_w[:, 2], min_delta_z)

        self._processed_actions[:] = delta_pos_w

        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        try:
            jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :][:, :, self._jacobi_joint_ids]
        except Exception as exc:
            raise RuntimeError(
                "CartesianDeltaAction failed to read robot Jacobian. "
                f"body={self._body_name}, body_idx={self._body_idx}, jacobi_body_idx={self._jacobi_body_idx}, "
                f"joint_names={self._joint_names}, joint_ids={self._joint_ids}"
            ) from exc
        if jacobian.shape[1] < 6 or jacobian.shape[2] != self._num_joints:
            raise RuntimeError(
                "CartesianDeltaAction got an invalid Jacobian shape: "
                f"shape={tuple(jacobian.shape)}, expected=(num_envs, 6, {self._num_joints})"
            )

        # ── Orientation: HOLD the full EE rotation captured at episode reset ──
        # Exactly-determined 6-DOF IK (6 joints, 6D task) so descent never trades
        # orientation for position. The old soft orientation_weight DLS only aligned
        # the approach axis (2-DOF) and left yaw unconstrained → crane drifted under
        # aggressive policy Z descent. See project-crane-fix Fix v6.
        R_cur = matrix_from_quat(self._asset.data.body_quat_w[:, self._body_idx])
        if self._target_rot_w is None or self._ori_captured is None:
            self._target_rot_w = R_cur.detach().clone()
            self._ori_captured = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            recap = ~self._ori_captured
            if bool(recap.any()):
                self._target_rot_w[recap] = R_cur[recap].detach()
                self._ori_captured[recap] = True
        R_d = self._target_rot_w
        # Siciliano orientation error: e_o = 0.5 Σ (col_cur × col_des)
        e_o = 0.5 * (
            torch.cross(R_cur[:, :, 0], R_d[:, :, 0], dim=1)
            + torch.cross(R_cur[:, :, 1], R_d[:, :, 1], dim=1)
            + torch.cross(R_cur[:, :, 2], R_d[:, :, 2], dim=1)
        )
        k_ori = float(self.cfg.orientation_gain)
        task = torch.cat([delta_pos_w, k_ori * e_o], dim=1)

        delta_q = dls_delta_q(jacobian, task, float(self.cfg.ik_damping))
        delta_q = torch.nan_to_num(delta_q, nan=0.0, posinf=0.0, neginf=0.0)

        # Speed limit via PROPORTIONAL scaling, not per-joint clip. A per-joint
        # clamp truncates only the joints that exceed the step limit, which
        # distorts the IK solution's position/orientation balance → crane drifts
        # during fast descent. Scaling the whole joint vector uniformly preserves
        # that balance (orientation stays held; descent just slows when limited).
        max_abs = delta_q.abs().amax(dim=1, keepdim=True).clamp(min=1.0e-9)
        speed_scale = (self._max_joint_step_rad / max_abs).clamp(max=1.0)
        delta_q = delta_q * speed_scale
        target = joint_pos + delta_q

        limits = joint_position_limits(self._asset.data)
        if limits is not None:
            lim = limits[:, self._joint_ids]
            target = torch.clamp(target, min=lim[..., 0], max=lim[..., 1])

        self._last_joint_delta[:] = target - joint_pos
        self._target_joint_pos[:] = target

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target_joint_pos, joint_ids=self._joint_ids)
        zero_vel = torch.zeros_like(self._target_joint_pos)
        self._asset.set_joint_velocity_target(zero_vel, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        # Mark orientation dirty so the next action step re-captures the reset EE
        # rotation as the hold target (camera down + reset yaw).
        if self._ori_captured is not None:
            self._ori_captured[env_ids] = False
        try:
            self._target_joint_pos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
            self._last_joint_delta[env_ids] = 0.0
        except Exception:
            pass


@configclass
class CartesianDeltaActionCfg(ActionTermCfg):
    """Policy-controlled XYZ action scaling, IK, and fixed Z floor."""

    class_type: type[ActionTerm] = CartesianDeltaAction
    joint_names: list[str] = MISSING
    body_name: str = MISSING
    position_scale_m: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_joint_speed_deg_s: float = 120.0
    ik_damping: float = 0.05
    preserve_order: bool = True
    # Legacy serialized fields: retained for config/snapshot compatibility. Full
    # rotation hold uses orientation_gain and the reset pose, not axis alignment.
    approach_axis: int = 2
    target_axis_w: tuple[float, float, float] = (0.0, 0.0, -1.0)
    z_target_w: float = 0.222
    orientation_weight: float = 0.35
    orientation_gain: float = 1.0
