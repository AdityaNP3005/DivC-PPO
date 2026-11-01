"""Full env configs: training (UR10eEnvCfg) and single-env scripted play (UR10eEnvCfg_PLAY)."""

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from config.task import (
    CARTESIAN_OBJECT_OFFSET_X_RANGE_M,
    CARTESIAN_OBJECT_OFFSET_Y_RANGE_M,
    CARTESIAN_OBJECT_OFFSET_Z_RANGE_M,
)
from rl_env.control import scripted_lift_success_termination
from rl_env.env.runtime import NUM_ENVS, PLAY_OBJECT_FARTHEST_RADIUS_M, REWARD_ERROR_SOURCE
from rl_env.env.scene_cfg import UR10eSceneCfg
from rl_env.env.observations_cfg import ActionsCfg, CommandsCfg, ObservationsCfg
from rl_env.env.events_cfg import EventCfg, _make_scripted_grasp_lift_event
from rl_env.env.rewards_cfg import RewardsCfg, TerminationsCfg


@configclass
class UR10eEnvCfg(ManagerBasedRLEnvCfg):
    """Config environment UR10e menggunakan USD custom.

    The robot is spawned per env from an instanceable USD (created by train_ur10e.py).
    The source USD is not modified.
    """

    scene: UR10eSceneCfg = UR10eSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2             # kontrol 30 Hz, fisika 60 Hz
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 1  # refresh randomized camera view before reset observations
        self.episode_length_s = 7.0    # 7 detik per episode
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.dt = 1.0 / 60.0

        # Training covers positioning only. The scripted grasp/lift runs only
        # in UR10eEnvCfg_PLAY, so the policy never learns from the gripper state.
        self.events.scripted_grasp_lift_control = None
        self.terminations.lift_success = None

        # The low-clearance penalty reads the GT cylinder top, so vision-reward runs drop it.
        # GT-reward runs (the paper's PPO and DivC-PPO) keep it, as in the paper checkpoints.
        # The fixed floor-clamp in CartesianDeltaAction already bounds the EE in Z.
        if REWARD_ERROR_SOURCE == "vision":
            self.rewards.low_depth = None

        # Cartesian policy uses reset-relative xyz + YOLO detection/error + last xyz action.
        self.events.reset_randomizable_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_randomizable_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.place_cylinder_under_gripper.params["x_offset_range_m"] = CARTESIAN_OBJECT_OFFSET_X_RANGE_M
        self.events.place_cylinder_under_gripper.params["y_offset_range_m"] = CARTESIAN_OBJECT_OFFSET_Y_RANGE_M
        self.events.place_cylinder_under_gripper.params["z_offset_range_m"] = CARTESIAN_OBJECT_OFFSET_Z_RANGE_M


@configclass
class UR10eEnvCfg_PLAY(UR10eEnvCfg):
    """Single-env play: positioning, then the scripted grasp/lift after stable success."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        # Nominal play keeps the validator's trained uniform-square placement. An explicit
        # PLAY_OBJECT_RADIUS_M remains available for visualising an OOD validation cell.
        if os.environ.get("PLAY_OBJECT_RADIUS_M", "").strip():
            self.events.place_cylinder_under_gripper.params["radial_offset_m"] = PLAY_OBJECT_FARTHEST_RADIUS_M

        # Play runs the rule-based grasp/lift after the stable-success stay.
        # Full 6-DOF IK controls wrist_2/wrist_3 (orientation hold) and the
        # scripted grasp controller owns the finger from the start → the
        # setpoint-hold event has nothing left to hold; disable it so it never
        # fights the IK or the grasp controller.
        self.events.hold_setpoint_only_joints = None
        self.events.scripted_grasp_lift_control = _make_scripted_grasp_lift_event()
        self.terminations.lift_success = DoneTerm(func=scripted_lift_success_termination)

        # Keep the object identical to validation during positioning. The scripted
        # weld moves the kinematic object with the EE after the grip is triggered.
