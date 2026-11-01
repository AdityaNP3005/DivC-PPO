"""Full env config used for training and evaluation (UR10eEnvCfg)."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from config.task import (
    CARTESIAN_OBJECT_OFFSET_X_RANGE_M,
    CARTESIAN_OBJECT_OFFSET_Y_RANGE_M,
    CARTESIAN_OBJECT_OFFSET_Z_RANGE_M,
)
from rl_env.env.runtime import NUM_ENVS, REWARD_ERROR_SOURCE
from rl_env.env.scene_cfg import UR10eSceneCfg
from rl_env.env.observations_cfg import ActionsCfg, CommandsCfg, ObservationsCfg
from rl_env.env.events_cfg import EventCfg
from rl_env.env.rewards_cfg import RewardsCfg, TerminationsCfg


@configclass
class UR10eEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e positioning environment built from the custom USD scene.

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
        self.decimation = 2             # 30 Hz control, 60 Hz physics
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 1  # refresh randomized camera view before reset observations
        self.episode_length_s = 7.0    # 7 s per episode
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.dt = 1.0 / 60.0

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
