"""Reset/interval events: joint resets, gripper hold, object placement."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from config.task import (
    CYLINDER_FLOOR_Z_M,
    OBJECT_CYL_HEIGHT_M,
    OBJECT_FLOOR_CLEARANCE_M,
)
from rl_env.control import (
    hold_joint_setpoints,
    place_cylinder_under_gripper,
    reset_joint_targets_to_positions,
)
from rl_env.env.runtime import (
    EE_LINK_NAME,
    GRIPPER_JOINT_NAMES_EXPR,
    JOINT_NAMES_FIXED_RESET_EXPR,
    JOINT_NAMES_RANDOM_RESET_EXPR,
    JOINT_NAMES_SETPOINT_ONLY_EXPR,
    JOINT_SETPOINT_ONLY_TARGETS,
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
