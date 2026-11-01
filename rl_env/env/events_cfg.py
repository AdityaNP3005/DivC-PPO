"""Reset/interval events: joint resets, gripper hold, object placement, scripted grasp (play)."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from config.task import (
    CYLINDER_FLOOR_Z_M,
    GRIPPER_CLOSE_MIMIC,
    GRIPPER_DRIVE_UNDERACTUATED,
    GRIPPER_FULL_CLOSE_RAD,
    GT_SUCCESS_XY_TOL_M,
    GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
    OBJECT_CYL_HEIGHT_M,
    OBJECT_FLOOR_CLEARANCE_M,
    SCRIPTED_GRIPPER_CLOSE_STEPS,
    SCRIPTED_GRIPPER_CLOSE_TARGETS,
    SCRIPTED_GRIPPER_HOLD_STEPS,
    SCRIPTED_IK_MAX_JOINT_STEP_RAD,
    SCRIPTED_LIFT_EXTRA_HEIGHT_M,
    SCRIPTED_LIFT_HEIGHT_M,
    SCRIPTED_LIFT_HOLD_STEPS,
    SCRIPTED_LIFT_SPEED_M_S,
    SCRIPTED_LIFT_SUCCESS_TOL_M,
    SCRIPTED_STAY_STEPS_REQUIRED,
    SCRIPTED_STAY_Z_TOL_M,
    SCRIPTED_WELD_ON_GRASP,
)
from rl_env.control import (
    hold_joint_setpoints,
    place_cylinder_under_gripper,
    reset_joint_targets_to_positions,
    scripted_grasp_lift_control,
)
from rl_env.env.runtime import (
    EE_LINK_NAME,
    GRIPPER_JOINT_NAMES_EXPR,
    JOINT_NAMES_ACTION_EXPR,
    JOINT_NAMES_FIXED_RESET_EXPR,
    JOINT_NAMES_RANDOM_RESET_EXPR,
    JOINT_NAMES_SETPOINT_ONLY_EXPR,
    JOINT_SETPOINT_ONLY_TARGETS,
)


def _make_scripted_grasp_lift_event() -> EventTerm:
    return EventTerm(
        func=scripted_grasp_lift_control,
        mode="interval",
        interval_range_s=(1.0 / 60.0, 1.0 / 60.0),
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME]),
            "object_cfg": SceneEntityCfg("cylinder"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES_EXPR),
            "action_joint_names": JOINT_NAMES_ACTION_EXPR,
            "object_height_m": OBJECT_CYL_HEIGHT_M,
            "target_clearance_m": GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
            "xy_tol_m": GT_SUCCESS_XY_TOL_M,
            "z_tol_m": SCRIPTED_STAY_Z_TOL_M,
            "stay_steps_required": SCRIPTED_STAY_STEPS_REQUIRED,
            "close_steps": SCRIPTED_GRIPPER_CLOSE_STEPS,
            "hold_after_close_steps": SCRIPTED_GRIPPER_HOLD_STEPS,
            "gripper_close_targets": SCRIPTED_GRIPPER_CLOSE_TARGETS,
            "gripper_drive_underactuated": GRIPPER_DRIVE_UNDERACTUATED,
            "gripper_full_close_rad": GRIPPER_FULL_CLOSE_RAD,
            "gripper_close_mimic": GRIPPER_CLOSE_MIMIC,
            "weld_on_grasp": SCRIPTED_WELD_ON_GRASP,
            "lift_height_m": SCRIPTED_LIFT_HEIGHT_M,
            "lift_hold_steps": SCRIPTED_LIFT_HOLD_STEPS,
            "lift_success_tol_m": SCRIPTED_LIFT_SUCCESS_TOL_M,
            "lift_extra_height_m": SCRIPTED_LIFT_EXTRA_HEIGHT_M,
            "lift_speed_m_s": SCRIPTED_LIFT_SPEED_M_S,
            "max_joint_step_rad": SCRIPTED_IK_MAX_JOINT_STEP_RAD,
        },
    )


@configclass
class EventCfg:
    """Reset the joints to the same initial position every episode.

    Uses reset_joints_by_offset with a zero range so the fixed joints return
    to their default/init_state values without randomization. Other joints can still
    be randomized by a separate reset event.
    """

    reset_randomizable_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_RANDOM_RESET_EXPR),
        },
    )

    reset_fixed_wrist_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_FIXED_RESET_EXPR),
        },
    )

    reset_gripper_joint = EventTerm(
        func=reset_joint_targets_to_positions,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES_EXPR),
            "joint_targets": {"finger_joint": 0.0},
            "zero_velocity": True,
        },
    )

    # Hold the non-RL joints at fixed setpoints every control step.
    # These joints can be set but are not part of the policy action.
    hold_setpoint_only_joints = EventTerm(
        func=hold_joint_setpoints,
        mode="interval",
        # Hold only the gripper; all arm joints belong to Cartesian IK.
        interval_range_s=(1.0 / 60.0, 1.0 / 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_SETPOINT_ONLY_EXPR),
            "joint_targets": JOINT_SETPOINT_ONLY_TARGETS,
            "zero_velocity": True,
        },
    )

    scripted_grasp_lift_control = _make_scripted_grasp_lift_event()

    place_cylinder_under_gripper = EventTerm(
        func=place_cylinder_under_gripper,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME]),
            "cylinder_cfg": SceneEntityCfg("cylinder"),
            "bracket_cfg": None,
            "floor_z_m": 0.0,
            "object_height_m": OBJECT_CYL_HEIGHT_M,
            "clearance_m": OBJECT_FLOOR_CLEARANCE_M,
            "cylinder_floor_z_m": CYLINDER_FLOOR_Z_M,
            "x_offset_range_m": (0.00, 0.00),
            "y_offset_range_m": (0.00, 0.00),
            "z_offset_range_m": (0.00, 0.00),
            "ee_local_offset_m": (0.05, -0.05, 0.0),
        },
    )
