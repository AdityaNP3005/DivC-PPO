from __future__ import annotations

"""Control package: cartesian IK action + joint setpoints + reset-time object
placement + scripted grasp/lift.

    cartesian_ik       — CartesianDeltaAction(+Cfg): 6-DOF orientation-hold DLS IK,
                         plus the shared DLS solve / Jacobian index / joint-limit helpers
    joint_setpoints    — reset_joint_targets_to_positions, hold_joint_setpoints
    cylinder_placement — place_cylinder_under_gripper (reset-time XY randomization)
    scripted_grasp     — scripted_grasp_lift_control state machine (play only)

This package owns Cartesian action, joint setpoints and scripted evaluation control.
"""

from rl_env.control.cartesian_ik import CartesianDeltaAction, CartesianDeltaActionCfg
from rl_env.control.joint_setpoints import (
    hold_joint_setpoints,
    reset_joint_targets_to_positions,
)
from rl_env.control.cylinder_placement import place_cylinder_under_gripper
from rl_env.control.scripted_grasp import (
    scripted_grasp_lift_control,
    scripted_lift_success_termination,
)

__all__ = [
    "CartesianDeltaAction",
    "CartesianDeltaActionCfg",
    "hold_joint_setpoints",
    "reset_joint_targets_to_positions",
    "place_cylinder_under_gripper",
    "scripted_grasp_lift_control",
    "scripted_lift_success_termination",
]
