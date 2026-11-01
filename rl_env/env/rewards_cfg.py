"""Reward + termination wiring. Weights/scales: config/rewards_weights.py; math: rl_env/rewards/.

Term names are stable: TensorBoard tags and rl_env/policy/axis_rewards.py key on them.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import config.rewards_weights as rw
from config.task import (
    GT_MIN_CLEARANCE_ABOVE_TOP_M,
    GT_SUCCESS_XY_TOL_M,
    GT_SUCCESS_Z_TOL_M,
    GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
    OBJECT_CYL_HEIGHT_M,
    YOLO_CONFIDENCE_THRESHOLD_PENALTY,
)
from rl_env.env.runtime import (
    EE_LINK_NAME,
    JOINT_NAMES_ACTION_EXPR,
    JOINT_NAMES_CLAMP_EXPR,
    REWARD_ERROR_SOURCE,
)
from rl_env.rewards.safety_terms import action_rate_l2_axis, joint_limit_clamp_penalty, yolo_no_detection_penalty
from rl_env.rewards.target_terms import (
    gt_low_clearance_penalty,
    target_abs_penalty,
    target_alive_penalty,
    target_exp_reward,
    target_idle_penalty,
    target_progress_reward,
    target_stay_reward,
    target_success_once_bonus,
)


def gt_target_params(ee: bool = True, **extra) -> dict:
    """Fresh EE/object entities + GT hover-target geometry (rewards and critic observations)."""
    params = {"ee_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME])} if ee else {}
    params.update(
        object_cfg=SceneEntityCfg("cylinder"),
        object_height_m=OBJECT_CYL_HEIGHT_M,
        clearance_above_m=GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
    )
    params.update(extra)
    return params


def _target_params(**overrides) -> dict:
    """Target-term params: GT target geometry + error source; term-specific values on top."""
    return gt_target_params(error_source=REWARD_ERROR_SOURCE, **overrides)


def _target_term(func, weight: float, **params) -> RewTerm:
    return RewTerm(func=func, weight=weight, params=_target_params(**params))


_NORM = dict(xy_scale_m=rw.ERROR_NORM_XY_SCALE_M, z_scale_m=rw.ERROR_NORM_Z_SCALE_M)
_BAND = dict(xy_tol_m=GT_SUCCESS_XY_TOL_M, z_tol_m=GT_SUCCESS_Z_TOL_M)
_IDLE = dict(
    idle_xy_threshold_m=GT_SUCCESS_XY_TOL_M,
    idle_z_threshold_m=GT_SUCCESS_Z_TOL_M,
    idle_action_threshold=rw.IDLE_ACTION_THRESHOLD,
    action_term_name="arm_action",
)


@configclass
class RewardsCfg:
    """Per-axis shaping over the configured GT or vision error source (see docs/REWARD_DESIGN.md)."""

    # Best-so-far progress per axis.
    penalty_x = _target_term(target_progress_reward, rw.W_PROGRESS_X, active_axes="x", **_NORM)
    penalty_y = _target_term(target_progress_reward, rw.W_PROGRESS_Y, active_axes="y", **_NORM)
    penalty_z = _target_term(target_progress_reward, rw.W_PROGRESS_Z, active_axes="z", **_NORM)
    # Bounded normalized distance per axis.
    abs_x = _target_term(target_abs_penalty, rw.W_ABS_X, active_axes="x", **_NORM)
    abs_y = _target_term(target_abs_penalty, rw.W_ABS_Y, active_axes="y", **_NORM)
    abs_z = _target_term(
        target_abs_penalty, rw.W_ABS_Z, active_axes="z", xy_scale_m=rw.ERROR_NORM_XY_SCALE_M, z_scale_m=rw.ABS_Z_SCALE_M
    )
    # Exponential, coarse scale: far-field pull.
    exp_x_coarse = _target_term(target_exp_reward, rw.W_EXP_COARSE_X, active_axes="x", xy_scale_m=rw.EXP_COARSE_XY_SCALE_M)
    exp_y_coarse = _target_term(target_exp_reward, rw.W_EXP_COARSE_Y, active_axes="y", xy_scale_m=rw.EXP_COARSE_XY_SCALE_M)
    exp_z_coarse = _target_term(target_exp_reward, rw.W_EXP_COARSE_Z, active_axes="z", z_scale_m=rw.EXP_COARSE_Z_SCALE_M)
    # Exponential, fine scale: precision in the last few cm.
    exp_x = _target_term(target_exp_reward, rw.W_EXP_FINE_X, active_axes="x", xy_scale_m=rw.EXP_FINE_XY_SCALE_M)
    exp_y = _target_term(target_exp_reward, rw.W_EXP_FINE_Y, active_axes="y", xy_scale_m=rw.EXP_FINE_XY_SCALE_M)
    exp_z = _target_term(target_exp_reward, rw.W_EXP_FINE_Z, active_axes="z", z_scale_m=rw.EXP_FINE_Z_SCALE_M)
    # GT-only safety near the cylinder top; assembly removes it in pure-vision runs.
    low_depth = RewTerm(
        func=gt_low_clearance_penalty,
        weight=rw.W_LOW_CLEARANCE,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME]),
            "object_cfg": SceneEntityCfg("cylinder"),
            "object_height_m": OBJECT_CYL_HEIGHT_M,
            "min_clearance_above_top_m": GT_MIN_CLEARANCE_ABOVE_TOP_M,
        },
    )
    # Off target while the axis action is ~zero.
    gt_idle_x = _target_term(target_idle_penalty, rw.W_IDLE, active_axes="x", **_IDLE)
    gt_idle_y = _target_term(target_idle_penalty, rw.W_IDLE, active_axes="y", **_IDLE)
    gt_idle_z = _target_term(target_idle_penalty, rw.W_IDLE, active_axes="z", **_IDLE)
    # Target band (all three axes): first entry, staying, time pressure outside it.
    success_once = _target_term(target_success_once_bonus, rw.W_SUCCESS_ONCE, **_BAND)
    stay_xy = _target_term(target_stay_reward, rw.W_STAY, **_BAND)
    alive_penalty = _target_term(target_alive_penalty, rw.W_ALIVE, **_BAND)
    # Safety / smoothness.
    no_detection_penalty = RewTerm(
        func=yolo_no_detection_penalty,
        weight=rw.W_NO_DET,
        params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "conf_threshold": YOLO_CONFIDENCE_THRESHOLD_PENALTY},
    )
    action_rate_x = RewTerm(func=action_rate_l2_axis, weight=rw.W_ACTION_RATE, params={"axis": 0})
    action_rate_y = RewTerm(func=action_rate_l2_axis, weight=rw.W_ACTION_RATE, params={"axis": 1})
    action_rate_z = RewTerm(func=action_rate_l2_axis, weight=rw.W_ACTION_RATE, params={"axis": 2})
    joint_limit_clamp = RewTerm(
        func=joint_limit_clamp_penalty,
        weight=rw.W_JOINT_CLAMP,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_CLAMP_EXPR),
            "max_deviation": rw.JOINT_MAX_DEVIATION_RAD,
            "action_term_name": "arm_action",
            "action_joint_names": JOINT_NAMES_ACTION_EXPR,
        },
    )


@configclass
class TerminationsCfg:
    """Episodes end only at the time limit."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


def _print_reward_terms() -> None:
    print(f"[env_cfg] Reward terms (error_source={REWARD_ERROR_SOURCE}):")
    for name, term in vars(RewardsCfg()).items():
        off = term.weight == 0 or (name == "low_depth" and REWARD_ERROR_SOURCE == "vision")
        print(f"[env_cfg]   [{'OFF ' if off else 'USED'}] {name:<22} weight={float(term.weight):>9g} | {term.func.__name__}")


_print_reward_terms()
