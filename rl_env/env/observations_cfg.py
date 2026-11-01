"""Action + observation wiring. Perception functions: rl_env/perception/obs_functions.py."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from config import BBOX_TARGET_CENTER_PX, BBOX_TARGET_DEPTH_M, CARTESIAN_ACTION_DEFAULTS
from config.task import (
    APPROACH_AXIS,
    CRANE_TARGET_AXIS_W,
    CYLINDER_FLOOR_Z_M,
    GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
    OBJECT_CYL_HEIGHT_M,
    OBJECT_FLOOR_CLEARANCE_M,
)
from rl_env.control import CartesianDeltaActionCfg
from rl_env.env.rewards_cfg import gt_target_params
from rl_env.env.runtime import _OBS_LAYOUT, ASYMMETRIC_CRITIC, EE_LINK_NAME, JOINT_NAMES_EXPR
from rl_env.perception import obs_functions as perception_obs
from rl_env.rewards import error_sources


@configclass
class CommandsCfg:
    """Target the robot has to reach.

    The setpoint is the GT target xyz.
    The 3-D command pose is not used.
    """

    # The old (x, y, z) setpoint is unused; the command stays idle.
    # Rewards and observations use the GT target xyz.


@configclass
class ActionsCfg:
    """Cartesian XYZ deltas converted to six-joint orientation-hold IK."""

    arm_action = CartesianDeltaActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_EXPR,   # full 6-DOF IK: holds reset orientation while policy commands XYZ
        body_name=EE_LINK_NAME,
        position_scale_m=tuple(CARTESIAN_ACTION_DEFAULTS["position_scale_m"]),
        max_joint_speed_deg_s=float(CARTESIAN_ACTION_DEFAULTS["max_joint_speed_deg_s"]),
        ik_damping=float(CARTESIAN_ACTION_DEFAULTS["ik_damping"]),
        approach_axis=APPROACH_AXIS,
        target_axis_w=CRANE_TARGET_AXIS_W,
        orientation_weight=2.0,   # vestigial (full 6-DOF IK ignores it)
        orientation_gain=float(CARTESIAN_ACTION_DEFAULTS["orientation_gain"]),
        preserve_order=True,
        z_target_w=(
            CYLINDER_FLOOR_Z_M
            + OBJECT_CYL_HEIGHT_M
            + OBJECT_FLOOR_CLEARANCE_M
            + GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M
        ),
    )


@configclass
class ObservationsCfg:
    """Deployable visual/proprioceptive observations with an optional GT critic tail."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Current EE xyz in raw meters relative to the configured initial FK pose.
        ee_delta_pos_obs = ObsTerm(
            func=perception_obs.ee_delta_pos_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME])},
        )
        # YOLO detection obs: center_x, center_y, depth (normalized [0,1]).
        yolo_detection_obs = ObsTerm(
            func=perception_obs.yolo_detection_obs,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )
        # Signed YOLO error: detection - visual target.
        yolo_detection_error_obs = ObsTerm(
            func=perception_obs.yolo_detection_error_obs,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "target_center_px": BBOX_TARGET_CENTER_PX,
                "target_depth_m": BBOX_TARGET_DEPTH_M,
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        # Critic-only privileged tail (6D): GT target_xyz + GT error_xyz.
        # The critic also reads ee_delta from the actor section [0:3] (not zeroed).
        # Actor width follows obs_layout (12D default, 10D legacy); GT tail is masked.
        critic_gt_target_pos_obs = ObsTerm(
            func=error_sources.gt_target_pos_obs,
            params=gt_target_params(ee=False),
        )
        critic_gt_error_obs = ObsTerm(
            func=error_sources.gt_ee_target_error_obs,
            params=gt_target_params(),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            if not ASYMMETRIC_CRITIC:
                # Non-asymmetric (pure-vision) ablation: no GT privileged tail.
                self.critic_gt_target_pos_obs = None
                self.critic_gt_error_obs = None
            # Preserve the depth channels of supported single-actor checkpoints.
            if _OBS_LAYOUT.depth_in_obs:
                self.yolo_detection_obs.params["include_depth"] = True
                self.yolo_detection_error_obs.params["include_depth"] = True

    policy: PolicyCfg = PolicyCfg()
