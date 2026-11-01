"""Lazy env configuration API; importing an env submodule does not assemble a scene.

Keep config's asset/default bootstrap eager and ordered before runtime constants
are requested. Framework utilities belong to their defining modules, not this API.
"""
from importlib import import_module

_CONFIG_EXPORTS = {
    "UR10eSceneCfg": "scene_cfg",
    "ActionsCfg": "observations_cfg",
    "CommandsCfg": "observations_cfg",
    "ObservationsCfg": "observations_cfg",
    "EventCfg": "events_cfg",
    "RewardsCfg": "rewards_cfg",
    "TerminationsCfg": "rewards_cfg",
    "UR10eEnvCfg": "assembly",
}
_RUNTIME_EXPORTS = (
    "ACTOR_DEPTH_ENABLED", "APPROACH_AXIS", "ASYMMETRIC_CRITIC",
    "ASYMMETRIC_CRITIC_DEFAULTS", "BBOX_TARGET_CENTER_PX", "BBOX_TARGET_DEPTH_M",
    "CAMERA_HEIGHT", "CAMERA_PRIM_PATH", "CAMERA_WIDTH",
    "CARTESIAN_ACTION_DEFAULTS", "CARTESIAN_OBJECT_OFFSET_X_RANGE_M", "CARTESIAN_OBJECT_OFFSET_Y_RANGE_M",
    "CARTESIAN_OBJECT_OFFSET_Z_RANGE_M", "CONTROL_MODE", "CRANE_TARGET_AXIS_W",
    "CUSTOM_SCENE_USD", "CYLINDER_FLOOR_Z_M", "EE_LINK_NAME",
    "EE_LINK_NAME_CANDIDATES", "GRIPPER_DISTAL_JOINTS_EXPR",
    "GRIPPER_DRIVE_DAMPING", "GRIPPER_DRIVE_STIFFNESS", 
    "GRIPPER_JOINT_NAME", "GRIPPER_JOINT_NAMES_EXPR",
    "GT_MIN_CLEARANCE_ABOVE_TOP_M", "GT_PRIVILEGED_OBS_DIM", "GT_SUCCESS_XY_TOL_M",
    "GT_SUCCESS_Z_TOL_M", "GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M", "JOINT_NAMES_ACTION_EXPR",
    "JOINT_NAMES_CLAMP_EXPR", "JOINT_NAMES_EXPR", "JOINT_NAMES_FIXED_RESET_EXPR",
    "JOINT_NAMES_RANDOM_RESET_EXPR", "JOINT_NAMES_SETPOINT_ONLY_EXPR", "JOINT_SETPOINT_ONLY_TARGETS",
    "NUM_ENVS", "OBJECT_CYL_HEIGHT_M", "OBJECT_CYL_MASS_KG",
    "OBJECT_CYL_RADIUS_M", "OBJECT_FLOOR_CLEARANCE_M", "OBJECT_USD",
    "OBJECT_USD_SCALE", "PLAY_OBJECT_FARTHEST_RADIUS_M", "REWARD_ERROR_SOURCE",
    "ROBOT_PRIM_PATH", 
    
    
    "SCRIPTED_STAY_STEPS_REQUIRED", 
    "YOLO_CONFIDENCE_THRESHOLD_PENALTY",
)
__all__ = [*_CONFIG_EXPORTS, *_RUNTIME_EXPORTS]


def __getattr__(name):
    module = _CONFIG_EXPORTS.get(name)
    if module is None:
        if name not in _RUNTIME_EXPORTS:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module = "runtime"
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
