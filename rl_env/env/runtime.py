"""Robot/scene constants and run-mode flags for the env configs (no Isaac imports).

Defines the UR10e prim/EE/joint names and the run-mode flags read from the environment
published by config.policy.publish_run_mode. It also re-exports the config values that
rl_env.env exposes lazily (the public `from rl_env.env import X` API); the env config
modules import framework classes and config values from their defining modules.
"""

# Public rl_env.env API: re-exported config values (see rl_env/env/__init__.py).
from config import (  # noqa: F401
    ASYMMETRIC_CRITIC_DEFAULTS,
    BBOX_TARGET_CENTER_PX,
    BBOX_TARGET_DEPTH_M,
    CAMERA_HEIGHT,
    CAMERA_PRIM_PATH,
    CAMERA_WIDTH,
    CARTESIAN_ACTION_DEFAULTS,
)
from config.task import (  # noqa: F401
    APPROACH_AXIS,
    CARTESIAN_OBJECT_OFFSET_X_RANGE_M,
    CARTESIAN_OBJECT_OFFSET_Y_RANGE_M,
    CARTESIAN_OBJECT_OFFSET_Z_RANGE_M,
    CRANE_TARGET_AXIS_W,
    CYLINDER_FLOOR_Z_M,
    GRIPPER_DISTAL_JOINTS_EXPR,
    GRIPPER_DRIVE_DAMPING,
    GRIPPER_DRIVE_STIFFNESS,
    GT_MIN_CLEARANCE_ABOVE_TOP_M,
    GT_SUCCESS_XY_TOL_M,
    GT_SUCCESS_Z_TOL_M,
    GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
    OBJECT_CYL_HEIGHT_M,
    OBJECT_CYL_MASS_KG,
    OBJECT_CYL_RADIUS_M,
    OBJECT_FLOOR_CLEARANCE_M,
    OBJECT_USD,
    OBJECT_USD_SCALE,
    SCRIPTED_STAY_STEPS_REQUIRED,
    YOLO_CONFIDENCE_THRESHOLD_PENALTY,
)
from config.control import require_cartesian_mode
from config.env_flags import env_bool, env_float
from config.reward_units import reward_error_source
from rl_env.obs_layout import resolve_obs_layout
from rl_env.project_config import CUSTOM_SCENE_USD  # noqa: F401

# =============================================================================
# CONFIGURATION — MUST MATCH THE USD SCENE
# =============================================================================

# Robot prim path per env (IsaacLab expands env_.* to env_0, env_1, ...)
ROBOT_PRIM_PATH = "/World/envs/env_.*/ur10e"

# End-effector link used for reward tracking
# ee_link is a massless fixed-joint link that PhysX merges into wrist_3_link, so it is not a body.
# Gripper EE body = the GRIPPER base, confirmed from the runtime articulation body list:
#   2F-140 → 'robotiq_base_link'  (2F-85 → 'base_link_0').
# NOTE: 'base_link' in that list is the UR10e ARM base (the fixed bottom link), NOT the EE —
# do NOT use it. The UR10e payload is referenced over HTTPS so pxr Stage.Open (used by
# _patch_gripper_fallbacks) cannot validate these as rigid bodies; the patcher then falls
# back to EE_LINK_NAME_CANDIDATES[0]. So candidates[0] MUST be the correct runtime EE name.
EE_LINK_NAME_CANDIDATES = ("robotiq_base_link", "base_link_0")
EE_LINK_NAME = EE_LINK_NAME_CANDIDATES[0]

# UR10e arm joints in kinematic order (not gripper/mimic joints). Every arm joint
# group below is a slice of this one list.
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Policy commands XYZ; orientation-hold IK owns all six arm joints.
JOINT_NAMES_EXPR = list(ARM_JOINT_NAMES)
# Shoulder, elbow and wrist_1: the joints checked by the joint-deviation penalty.
JOINT_NAMES_ACTION_EXPR = list(ARM_JOINT_NAMES[:4])
# Arm reset groups preserve the initial pose used by Cartesian IK.
JOINT_NAMES_RANDOM_RESET_EXPR = list(ARM_JOINT_NAMES[:4])
# Joint-limit clamp penalty covers the policy-moved shoulder/elbow/wrist_1 joints.
JOINT_NAMES_CLAMP_EXPR = list(ARM_JOINT_NAMES[1:4])

# Robotiq 2F-140 main finger joint: [0.0 open .. 0.7854 rad = 45° closed]. The USD
# mimics the other fingers from it, so the gripper command is this one joint.
GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_JOINT_NAMES_EXPR = [GRIPPER_JOINT_NAME]
# Remaining joints reset to the same initial pose every episode.
JOINT_NAMES_FIXED_RESET_EXPR = list(ARM_JOINT_NAMES[4:]) + [GRIPPER_JOINT_NAME]

# Setpoint-only (non-policy) joints are re-targeted every control step to hold the gripper open.
JOINT_NAMES_SETPOINT_ONLY_EXPR = [GRIPPER_JOINT_NAME]
JOINT_SETPOINT_ONLY_TARGETS = {GRIPPER_JOINT_NAME: 0.0}

# Number of parallel envs (train_ur10e.py builds the instanceable robot USD at runtime)
NUM_ENVS = 128

# Object spec + GT target geometry/tolerances live in config/task.py (imported at top);
# reward weights/scales in config/rewards_weights.py (imported by rewards_cfg.py).
GT_PRIVILEGED_OBS_DIM = int(ASYMMETRIC_CRITIC_DEFAULTS.get("privileged_obs_dim", 6))

# =============================================================================
# VISION PARAMETERS — YOLO detection threshold
# =============================================================================

# Camera + vision target (CAMERA_*, BBOX_TARGET_*) live in config/perception.py,
# the single source for training and validation. Imported above.
# Task constants (tolerances, object, gripper drive, approach axis) live in config/task.py.

# Cartesian mode: current xyz observation is raw meters relative to the configured initial FK pose.
# Randomization is applied as target/object xyz offset, not as random joint reset.
# CARTESIAN_OBJECT_OFFSET_{X,Y,Z}_RANGE_M now live in config/task.py (imported at top).
# Out-of-distribution placement (PLAY_OBJECT_RADIUS_M): instead of per-axis uniform sampling, place
# the object at the FARTHEST trained distance (fixed radius = max per-axis magnitude) but a
# RANDOM direction each reset. The radius equals the per-axis half-range so it stays inside
# the trained range (the inscribed circle), exercising the policy at the edge from every angle.
# Env-overridable (PLAY_OBJECT_RADIUS_M) so an eval can push the object OUTSIDE the
# trained square — the out-of-distribution placement test (Table V). Unset → the
# trained edge radius, so every other path is unchanged.
PLAY_OBJECT_FARTHEST_RADIUS_M = env_float("PLAY_OBJECT_RADIUS_M", max(
    abs(CARTESIAN_OBJECT_OFFSET_X_RANGE_M[0]), abs(CARTESIAN_OBJECT_OFFSET_X_RANGE_M[1]),
    abs(CARTESIAN_OBJECT_OFFSET_Y_RANGE_M[0]), abs(CARTESIAN_OBJECT_OFFSET_Y_RANGE_M[1]),
))

# =============================================================================
# RUN MODE — control mode, critic and reward error source
# =============================================================================
import os as _os
_MODEL_TYPE = "PPO"
CONTROL_MODE = require_cartesian_mode(_os.environ.get("RL_CONTROL_MODE"))

# Published by config.policy.publish_run_mode (training and validation).
ASYMMETRIC_CRITIC = env_bool("ASYMMETRIC_CRITIC", False)
ACTOR_DEPTH_ENABLED = env_bool("ACTOR_DEPTH_ENABLED", True)
_OBS_LAYOUT = resolve_obs_layout(ASYMMETRIC_CRITIC, ACTOR_DEPTH_ENABLED)

REWARD_ERROR_SOURCE = reward_error_source(ASYMMETRIC_CRITIC)

print(
    f"[env_cfg] Active: {_MODEL_TYPE} | mode={CONTROL_MODE} | "
    f"asym_critic={ASYMMETRIC_CRITIC} | reward_error_source={REWARD_ERROR_SOURCE}"
)
