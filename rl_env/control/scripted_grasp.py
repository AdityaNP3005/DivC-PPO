from __future__ import annotations

"""Scripted grasp + lift controller (fixed, PLAY-only).

Phase machine: 0 = RL approach/stay, 1 = manual gripper close, 2 = 6-DOF IK lift,
3 = lifted/success. Drives the underactuated 2F-140, optionally welds the object to
the EE, and runs a slow damped-least-squares IK lift while holding EE orientation.
"""

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from rl_env.control.cartesian_ik import dls_delta_q, jacobian_body_index, joint_position_limits
from rl_env.env_ids import env_ids_tensor as _env_ids_tensor, just_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_grasp_lift_state(env: "ManagerBasedRLEnv") -> None:
    """Allocate per-env scripted grasp/lift buffers.

    Phase convention:
    0 = RL approach/stay, 1 = manual gripper closing, 2 = IK lift, 3 = lifted/success.
    """
    if hasattr(env, "_scripted_grasp_phase"):
        return
    env._scripted_grasp_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._scripted_stay_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._scripted_phase_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._scripted_lift_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env._scripted_object_lift_hold_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._scripted_lift_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    env._scripted_lift_target_pos_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    env._scripted_lift_start_quat_w = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)  # (w,x,y,z)
    env._scripted_object_start_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    env._scripted_debug_gripper_printed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Weld-on-grasp: once the grip triggers, the object is kinematically attached to the EE
    # (rigid relative transform captured at close) so the lift always succeeds despite the
    # underactuated gripper linkage only making tip contact. Play-only scripted demo.
    env._scripted_weld_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env._scripted_weld_obj_pos_b = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    env._scripted_weld_obj_quat_b = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
    # Lazily resized in scripted_grasp_lift_control once action_joint_names are known.
    env._scripted_action_hold_joint_pos = torch.zeros(env.num_envs, 0, device=env.device, dtype=torch.float32)


def _reset_grasp_lift_state_for_new_episodes(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    _ensure_grasp_lift_state(env)
    if env_ids.numel() == 0:
        return
    reset_mask = just_reset(env)[env_ids]
    reset_ids = env_ids[reset_mask]
    if reset_ids.numel() == 0:
        return
    env._scripted_grasp_phase[reset_ids] = 0
    env._scripted_stay_count[reset_ids] = 0
    env._scripted_phase_step[reset_ids] = 0
    env._scripted_lift_success[reset_ids] = False
    env._scripted_object_lift_hold_count[reset_ids] = 0
    env._scripted_lift_start_pos_w[reset_ids] = 0.0
    env._scripted_lift_target_pos_w[reset_ids] = 0.0
    env._scripted_lift_start_quat_w[reset_ids] = 0.0
    env._scripted_object_start_z[reset_ids] = 0.0
    env._scripted_debug_gripper_printed[reset_ids] = False
    if hasattr(env, "_scripted_weld_active"):
        env._scripted_weld_active[reset_ids] = False
        env._scripted_weld_obj_pos_b[reset_ids] = 0.0
        env._scripted_weld_obj_quat_b[reset_ids] = 0.0
    if hasattr(env, "_scripted_action_hold_joint_pos") and env._scripted_action_hold_joint_pos.numel() > 0:
        env._scripted_action_hold_joint_pos[reset_ids] = 0.0


def _gripper_close_multiplier(joint_name: str, mimic_map: dict[str, float]) -> float | None:
    """Map a joint name → its closed-angle multiplier (× full_close_rad).

    The driver is matched by EXACT name "finger_joint" (so "right_inner_finger_joint" does
    NOT accidentally get the driver's +1). All other finger DOFs match by name SUBSTRING.
    Returns None when the joint is not gripper-related (skip it).
    """
    low = joint_name.lower()
    if low == "finger_joint":
        return float(mimic_map.get("finger_joint", 1.0))
    # Substring keys, MOST-SPECIFIC (longest) first so "inner_finger_pad" wins over
    # "inner_finger", and the driver key "finger_joint" never matches by substring.
    for key in sorted((k for k in mimic_map if k != "finger_joint"), key=len, reverse=True):
        if key in low:
            return float(mimic_map[key])
    return None


def _resolve_gripper_drive_joints(env, robot, mimic_map: dict[str, float]):
    """Discover every finger DOF in the articulation + its close multiplier (cached on env).

    Returns (joint_ids_tensor, names, signs_tensor). Drives the underactuated 2F-140 pad
    even when the USD mimic constraint is not applied in sim. Also prints a one-time dump of
    ALL robot joints so the real names/signs in the USD can be confirmed and tuned.
    """
    cache = getattr(env, "_gripper_drive_cache", None)
    if cache is not None:
        return cache

    all_names = list(getattr(robot.data, "joint_names", []) or [])
    sel_ids: list[int] = []
    sel_names: list[str] = []
    sel_signs: list[float] = []
    for jname in all_names:
        mult = _gripper_close_multiplier(jname, mimic_map)
        if mult is None:
            continue
        jid, _ = robot.find_joints([jname], preserve_order=True)
        if len(jid) > 0:
            sel_ids.append(int(jid[0]))
            sel_names.append(jname)
            sel_signs.append(float(mult))

    ids_t = torch.tensor(sel_ids, device=env.device, dtype=torch.long) if sel_ids else torch.empty(0, device=env.device, dtype=torch.long)
    signs_t = torch.tensor(sel_signs, device=env.device, dtype=torch.float32) if sel_signs else torch.empty(0, device=env.device, dtype=torch.float32)
    cache = (ids_t, sel_names, signs_t)
    env._gripper_drive_cache = cache

    # One-time diagnostic: every joint + which were selected as gripper drive joints.
    if not getattr(env, "_gripper_joints_dumped", False):
        try:
            jp = robot.data.joint_pos[0].detach().cpu().tolist()
        except Exception:
            jp = [float("nan")] * len(all_names)
        try:
            jstiff = robot.data.joint_stiffness[0].detach().cpu().tolist()
            jdamp = robot.data.joint_damping[0].detach().cpu().tolist()
        except Exception:
            jstiff = jdamp = [float("nan")] * len(all_names)
        print("[SCRIPTED][joints] full articulation joint list (env 0):", flush=True)
        for i, nm in enumerate(all_names):
            tag = ""
            m = _gripper_close_multiplier(nm, mimic_map)
            if m is not None:
                tag = f"  <-- GRIPPER DRIVE (close mult={m:+.2f})"
            _kd = f" stiff={jstiff[i]:.1f} damp={jdamp[i]:.1f}" if i < len(jstiff) else ""
            print(f"    [{i:2d}] {nm:<32} pos={jp[i] if i < len(jp) else float('nan'):+.4f} rad{_kd}{tag}", flush=True)
        print("[SCRIPTED][gains] If a GRIPPER DRIVE joint shows stiff=0.0 it was PASSIVE and "
              "could not be commanded before the gripper_distal actuator was added.", flush=True)
        if not sel_names:
            print("[SCRIPTED][joints] WARNING: no finger DOFs matched — inner joints may be PhysX "
                  "mimic constraints (not DOFs). If the pad still won't wrap, the fix is at the USD "
                  "mimic level, not here.", flush=True)
        env._gripper_joints_dumped = True

    return cache


def scripted_grasp_lift_control(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | slice | None,
    robot_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    action_joint_names: list[str],
    object_height_m: float = 0.10,
    target_clearance_m: float = 0.12,
    xy_tol_m: float = 0.008,
    z_tol_m: float = 0.008,
    stay_steps_required: int = 20,
    close_steps: int = 20,
    hold_after_close_steps: int = 0,
    gripper_close_targets: dict[str, float] | None = None,
    gripper_drive_underactuated: bool = True,
    gripper_full_close_rad: float = 0.7854,
    gripper_close_mimic: dict[str, float] | None = None,
    weld_on_grasp: bool = True,
    lift_height_m: float = 0.15,
    lift_hold_steps: int = 20,
    lift_success_tol_m: float = 0.02,
    lift_extra_height_m: float = 0.15,
    lift_speed_m_s: float = 0.05,
    max_joint_step_rad: float = 0.012,
    ik_damping: float = 0.05,
) -> None:
    """Scripted post-stay grasp and lift controller.

    RL controls approach until GT target stay is stable for `stay_steps_required` steps.
    Then this function manually closes the gripper and overrides arm joint targets
    with a slow Jacobian IK lift. Gripper joints remain observable through the
    play-side scripted controller state.
    """
    from isaaclab.assets import Articulation, RigidObject

    ids = _env_ids_tensor(env, env_ids)
    if ids.numel() == 0:
        return
    _ensure_grasp_lift_state(env)
    _reset_grasp_lift_state_for_new_episodes(env, ids)

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    action_joint_ids, _ = robot.find_joints(list(action_joint_names), preserve_order=True)
    if (
        not hasattr(env, "_scripted_action_hold_joint_pos")
        or env._scripted_action_hold_joint_pos.shape != (env.num_envs, len(action_joint_ids))
    ):
        env._scripted_action_hold_joint_pos = torch.zeros(
            env.num_envs, len(action_joint_ids), device=env.device, dtype=robot.data.joint_pos.dtype
        )

    # Lazy: rl_env.rewards imports this package's siblings. The grasp trigger is always
    # GT, independent of the training reward's error source.
    from rl_env.rewards.target_terms import _at_target

    stay_mask_all = _at_target(
        env, ee_cfg, object_cfg, "gt", object_height_m, target_clearance_m, xy_tol_m, z_tol_m
    )

    phase = env._scripted_grasp_phase
    phase_ids = phase[ids]
    approach_ids = ids[phase_ids == 0]
    if approach_ids.numel() > 0:
        stayed = stay_mask_all[approach_ids]
        env._scripted_stay_count[approach_ids] = torch.where(
            stayed,
            env._scripted_stay_count[approach_ids] + 1,
            torch.zeros_like(env._scripted_stay_count[approach_ids]),
        )
        trigger_ids = approach_ids[env._scripted_stay_count[approach_ids] >= int(stay_steps_required)]
        if trigger_ids.numel() > 0:
            phase[trigger_ids] = 1
            env._scripted_phase_step[trigger_ids] = 0
            # Freeze the arm at the stable camera pose before closing the gripper.
            env._scripted_action_hold_joint_pos[trigger_ids] = robot.data.joint_pos[trigger_ids][:, action_joint_ids].detach().clone()
            # Weld-on-grasp: capture the object's pose RELATIVE to the EE now (object still
            # centred under the open gripper) and attach it. Through close + lift the object is
            # held rigidly at this offset, so the underactuated tip-only grip can't drop it.
            if weld_on_grasp:
                _ee_id = int(ee_cfg.body_ids[0])
                _ee_p = robot.data.body_pos_w[trigger_ids, _ee_id]
                _ee_q = robot.data.body_quat_w[trigger_ids, _ee_id]
                _ob_p = obj.data.root_pos_w[trigger_ids, :3]
                _ob_q = obj.data.root_quat_w[trigger_ids]
                _p_b, _q_b = subtract_frame_transforms(_ee_p, _ee_q, _ob_p, _ob_q)
                env._scripted_weld_obj_pos_b[trigger_ids] = _p_b
                env._scripted_weld_obj_quat_b[trigger_ids] = _q_b
                env._scripted_weld_active[trigger_ids] = True
            new_print_ids = trigger_ids[~env._scripted_debug_gripper_printed[trigger_ids]]
            if new_print_ids.numel() > 0:
                env._scripted_debug_gripper_printed[new_print_ids] = True
                print(
                    f"[SCRIPTED] gripper close triggered envs={new_print_ids.detach().cpu().tolist()} "
                    f"stay_steps={int(stay_steps_required)} target={gripper_close_targets or {'finger_joint': 0.7}}",
                    flush=True,
                )

    # Weld update: every step, drive each attached object to EE_pose ∘ captured_offset and
    # zero its velocity so it rigidly follows the gripper (kinematic attach, no PhysX joint).
    if weld_on_grasp and hasattr(env, "_scripted_weld_active"):
        weld_ids = ids[env._scripted_weld_active[ids]]
        if weld_ids.numel() > 0:
            _ee_id = int(ee_cfg.body_ids[0])
            _ee_p = robot.data.body_pos_w[weld_ids, _ee_id]
            _ee_q = robot.data.body_quat_w[weld_ids, _ee_id]
            _new_p, _new_q = combine_frame_transforms(
                _ee_p, _ee_q,
                env._scripted_weld_obj_pos_b[weld_ids],
                env._scripted_weld_obj_quat_b[weld_ids],
            )
            obj.write_root_pose_to_sim(torch.cat([_new_p, _new_q], dim=-1), env_ids=weld_ids)
            obj.write_root_velocity_to_sim(
                torch.zeros((weld_ids.numel(), 6), device=env.device, dtype=_new_p.dtype),
                env_ids=weld_ids,
            )

    # Manual gripper override for all play phases:
    # - phase 0: hold open from the beginning so the finger does not shake
    # - phase 1: ramp close over `close_steps`, then hold full close for
    #   `hold_after_close_steps` before lift
    gripper_ids = ids
    if gripper_ids.numel() > 0:
        # ── Pick the joints to drive + their per-joint CLOSED target ──
        # Underactuated mode (2F-140): drive EVERY finger DOF discovered by name, with the
        # inner pad counter-rotating (mimic -1) so the pad WRAPS the object instead of only
        # the fingertip touching. Legacy mode: drive only gripper_cfg joints to close_targets.
        legacy_targets = gripper_close_targets or {"finger_joint": 0.7}
        mimic_map = gripper_close_mimic or {"finger_joint": 1.0}
        drive_ids = None
        drive_names: list[str] = []
        closed_vals = None  # per-joint full-closed angle (rad), shape [n_joints]
        if gripper_drive_underactuated:
            _ids_t, _names, _signs = _resolve_gripper_drive_joints(env, robot, mimic_map)
            if _ids_t.numel() > 0:
                drive_ids = _ids_t
                drive_names = _names
                closed_vals = _signs.to(dtype=robot.data.joint_pos.dtype) * float(gripper_full_close_rad)
        if drive_ids is None:
            # Fallback: legacy finger-only drive (relies on USD mimic for the rest).
            joint_names = list(gripper_cfg.joint_names) if gripper_cfg.joint_names is not None else []
            if joint_names:
                _gid, _ = robot.find_joints(joint_names, preserve_order=True)
                drive_ids = torch.as_tensor(_gid, device=env.device, dtype=torch.long)
                drive_names = joint_names
                closed_vals = torch.tensor(
                    [float(legacy_targets.get(n, 0.7)) for n in joint_names],
                    device=env.device, dtype=robot.data.joint_pos.dtype,
                )

        if drive_ids is not None and drive_ids.numel() > 0:
            phase_for_gripper = phase[gripper_ids]
            close_mask = (phase_for_gripper >= 1).unsqueeze(1)        # [n_env,1]
            closing_mask = phase_for_gripper == 1
            close_denom = max(int(close_steps), 1)
            close_progress = torch.clamp(
                env._scripted_phase_step[gripper_ids].to(dtype=robot.data.joint_pos.dtype) / float(close_denom),
                0.0, 1.0,
            )
            # Phase 1 ramps open->close over `close_steps`; later phases keep full close.
            close_progress = torch.where(closing_mask, close_progress, torch.ones_like(close_progress)).unsqueeze(1)  # [n_env,1]

            # target = open(0) ramped toward each joint's closed angle, only once closing.
            ramp = closed_vals.unsqueeze(0) * close_progress          # [n_env, n_j]
            target = torch.where(close_mask, ramp, torch.zeros_like(ramp))

            joint_limits = joint_position_limits(robot.data)
            if joint_limits is not None:
                limits = joint_limits[gripper_ids][:, drive_ids, :].to(device=env.device, dtype=target.dtype)
                target = torch.clamp(target, min=limits[..., 0], max=limits[..., 1])
                if not getattr(env, "_scripted_gripper_target_logged", False):
                    for _c, _n in enumerate(drive_names):
                        _cv = float(closed_vals[_c].item())
                        _lo, _hi = float(limits[0, _c, 0]), float(limits[0, _c, 1])
                        _sent = min(max(_cv, _lo), _hi)
                        _warn = "  <-- CLAMPED (sign/limit mismatch? check USD)" if abs(_sent - _cv) > 1e-4 else ""
                        print(
                            f"[SCRIPTED] {_n:<28} close_target={_cv:+.4f} rad ({math.degrees(_cv):+.1f}°) "
                            f"| limit=[{_lo:+.4f},{_hi:+.4f}] | sent={_sent:+.4f} rad{_warn}",
                            flush=True,
                        )
                    env._scripted_gripper_target_logged = True
            robot.set_joint_position_target(target, joint_ids=drive_ids, env_ids=gripper_ids)
            robot.set_joint_velocity_target(torch.zeros_like(target), joint_ids=drive_ids, env_ids=gripper_ids)

            # Periodic motion check (env 0): are the driven joints ACTUALLY moving toward
            # their targets? If a joint's pos stays put while target!=pos → still passive.
            _e0 = int(gripper_ids[0].item())
            _step0 = int(env._scripted_phase_step[_e0].item())
            if bool((phase[_e0] == 1).item()) and _step0 % 20 == 0:
                _pos0 = robot.data.joint_pos[_e0, drive_ids].detach().cpu().tolist()
                _tgt0 = target[0].detach().cpu().tolist()
                _pairs = ", ".join(f"{nm.split('_joint')[0]}: pos={p:+.3f}->tgt={t:+.3f}"
                                   for nm, p, t in zip(drive_names, _pos0, _tgt0))
                print(f"[SCRIPTED][close step {_step0:3d}] {_pairs}", flush=True)

    closing_ids = ids[phase[ids] == 1]
    if closing_ids.numel() > 0:
        # Override all RL-controlled arm joints during grip wait so the wrist camera
        # stays at the verified stay pose and detection does not shake.
        hold_q = env._scripted_action_hold_joint_pos[closing_ids].to(dtype=robot.data.joint_pos.dtype)
        zero_vel = torch.zeros_like(hold_q)
        robot.write_joint_state_to_sim(hold_q, zero_vel, joint_ids=action_joint_ids, env_ids=closing_ids)
        robot.set_joint_position_target(hold_q, joint_ids=action_joint_ids, env_ids=closing_ids)
        robot.set_joint_velocity_target(zero_vel, joint_ids=action_joint_ids, env_ids=closing_ids)

        env._scripted_phase_step[closing_ids] += 1
        lift_start_delay_steps = int(close_steps) + int(hold_after_close_steps)
        lift_start_ids = closing_ids[env._scripted_phase_step[closing_ids] >= lift_start_delay_steps]
        if lift_start_ids.numel() > 0:
            phase[lift_start_ids] = 2
            env._scripted_phase_step[lift_start_ids] = 0
            print(f"[SCRIPTED] lift phase started envs={lift_start_ids.detach().cpu().tolist()}", flush=True)
            ee_pos = robot.data.body_pos_w[lift_start_ids, ee_cfg.body_ids[0]]
            ee_quat = robot.data.body_quat_w[lift_start_ids, ee_cfg.body_ids[0]]
            env._scripted_lift_start_pos_w[lift_start_ids] = ee_pos
            env._scripted_lift_start_quat_w[lift_start_ids] = ee_quat
            env._scripted_lift_target_pos_w[lift_start_ids] = ee_pos + torch.tensor(
                [0.0, 0.0, float(lift_height_m)], device=env.device, dtype=ee_pos.dtype
            )
            env._scripted_object_start_z[lift_start_ids] = obj.data.root_pos_w[lift_start_ids, 2]
            env._scripted_object_lift_hold_count[lift_start_ids] = 0

    lift_ids = ids[phase[ids] == 2]
    if lift_ids.numel() > 0:
        body_id = ee_cfg.body_ids[0]
        jacobi_body_idx = jacobian_body_index(robot, body_id)
        # Full 6×n Jacobian: 3 position + 3 rotation rows
        jacobian_6d = robot.root_physx_view.get_jacobians()[lift_ids, jacobi_body_idx, :6, :][:, :, action_joint_ids]

        ee_pos = robot.data.body_pos_w[lift_ids, body_id]
        ee_quat = robot.data.body_quat_w[lift_ids, body_id]  # (w, x, y, z)
        object_target_z = env._scripted_object_start_z[lift_ids] + float(lift_height_m)
        object_lift_z_threshold = object_target_z - abs(float(lift_success_tol_m))
        object_lifted_now = obj.data.root_pos_w[lift_ids, 2] >= object_lift_z_threshold

        # Do not stop just because the gripper/EE has moved 15 cm. Success is
        # based on the object's z lift. Until the object reaches the target and
        # holds it, keep a slowly rising EE target with a bounded extra margin.
        max_target_z = env._scripted_lift_start_pos_w[lift_ids, 2] + float(lift_height_m) + float(lift_extra_height_m)
        need_more_lift = ~object_lifted_now
        env._scripted_lift_target_pos_w[lift_ids, 2] = torch.where(
            need_more_lift,
            torch.minimum(env._scripted_lift_target_pos_w[lift_ids, 2] + max(float(lift_speed_m_s) * float(env.step_dt), 1e-5), max_target_z),
            env._scripted_lift_target_pos_w[lift_ids, 2],
        )

        # ── Position error (only Z rises, X/Y stay at start position) ─────────
        target_pos = env._scripted_lift_target_pos_w[lift_ids]
        pos_error = target_pos - ee_pos
        max_step = max(float(lift_speed_m_s) * float(env.step_dt), 1e-5)
        err_norm = torch.norm(pos_error, dim=1, keepdim=True).clamp_min(1e-6)
        step_vec = pos_error * torch.clamp(max_step / err_norm, max=1.0)

        # ── Orientation error: maintain initial EE orientation ────────────────
        # q_diff = q_desired * inv(q_current) → convert to axis-angle error
        q_start = env._scripted_lift_start_quat_w[lift_ids]  # desired orientation
        q_curr = ee_quat
        q_diff = quat_mul(q_start, quat_inv(q_curr))
        orient_error = axis_angle_from_quat(q_diff)  # (N, 3), magnitude = angle in rad
        # Scale orientation error down so it doesn't dominate position tracking
        orient_error = orient_error * 0.3

        # ── 6D damped least squares IK ────────────────────────────────────────
        error_6d = torch.cat([step_vec, orient_error], dim=1)  # (N, 6)
        delta_q = dls_delta_q(jacobian_6d, error_6d, float(ik_damping))  # (N, n_joints)
        delta_q = torch.clamp(delta_q, -float(max_joint_step_rad), float(max_joint_step_rad))

        joint_pos = robot.data.joint_pos[lift_ids][:, action_joint_ids]
        joint_des = joint_pos + delta_q
        zero_vel = torch.zeros_like(joint_des)
        # Interval events run after physics/reward. Writing the state makes the
        # scripted IK override effective even though the RL action term will set
        # arm targets again at the next control step. Delta is clamped so motion
        # remains slow and incremental.
        robot.write_joint_state_to_sim(joint_des, zero_vel, joint_ids=action_joint_ids, env_ids=lift_ids)
        robot.set_joint_position_target(joint_des, joint_ids=action_joint_ids, env_ids=lift_ids)
        robot.set_joint_velocity_target(zero_vel, joint_ids=action_joint_ids, env_ids=lift_ids)

        object_lifted = obj.data.root_pos_w[lift_ids, 2] >= (
            env._scripted_object_start_z[lift_ids] + float(lift_height_m) - abs(float(lift_success_tol_m))
        )
        env._scripted_object_lift_hold_count[lift_ids] = torch.where(
            object_lifted,
            env._scripted_object_lift_hold_count[lift_ids] + 1,
            torch.zeros_like(env._scripted_object_lift_hold_count[lift_ids]),
        )
        success_ids = lift_ids[env._scripted_object_lift_hold_count[lift_ids] >= int(lift_hold_steps)]
        if success_ids.numel() > 0:
            phase[success_ids] = 3
            env._scripted_lift_success[success_ids] = True

    robot.write_data_to_sim()


def scripted_lift_success_termination(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Terminate episode after scripted lift success."""
    _ensure_grasp_lift_state(env)
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    _reset_grasp_lift_state_for_new_episodes(env, ids)
    return env._scripted_lift_success.clone()
