"""Scene: per-env robot (patched USD), dome light, target object and wrist camera."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from config import CAMERA_HEIGHT, CAMERA_PRIM_PATH, CAMERA_WIDTH
from config.task import (
    GRIPPER_DISTAL_JOINTS_EXPR,
    GRIPPER_DRIVE_DAMPING,
    GRIPPER_DRIVE_STIFFNESS,
    OBJECT_CYL_HEIGHT_M,
    OBJECT_CYL_MASS_KG,
    OBJECT_CYL_RADIUS_M,
    OBJECT_USD,
    OBJECT_USD_SCALE,
)
from rl_env.env.runtime import GRIPPER_JOINT_NAMES_EXPR, JOINT_NAMES_EXPR, ROBOT_PRIM_PATH
from rl_env.project_config import CUSTOM_SCENE_USD


def _make_object_spawn():
    """Object spawn for the per-env target.

    Default (OBJECT_USD unset) → the primitive orange cylinder.
    OBJECT_USD set → reference that USD (another object; it needs a matching detector). Same kinematic+no-gravity props so the placement event
    fully controls its pose (no falling), identical to the cylinder.
    """
    if OBJECT_USD:
        return sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD,
            scale=(OBJECT_USD_SCALE, OBJECT_USD_SCALE, OBJECT_USD_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                max_depenetration_velocity=1.0,
            ),
            # Defensive: if a referenced USD ever ships an ArticulationRootAPI (Isaac factory/YCB
            # props did), it conflicts with spawning as a single RigidObject — disable it so the
            # object is a plain rigid body the placement event can pose. Our primitive USDs have none.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_CYL_MASS_KG),
        )
    return sim_utils.CylinderCfg(
        radius=OBJECT_CYL_RADIUS_M,
        height=OBJECT_CYL_HEIGHT_M,
        axis="Z",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.11583012342453003, 0.0),  # orange
            roughness=0.35,
            metallic=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,          # training: stable Z, does not fall
            kinematic_enabled=True,        # training: position locked
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_CYL_MASS_KG),
    )


@configclass
class UR10eSceneCfg(InteractiveSceneCfg):
    """Scene config for the UR10e.

    The robot is spawned per env by IsaacLab via UsdFileCfg (instanceable=True).
    train_ur10e.py membuat temp USD robot (ArticulationRoot fix + ee_link mass fix),
    then overrides robot.spawn.usd_path before gym.make().
    The environment (ground plane, lights, etc.) is loaded separately as a sublayer.
    The source USD is never modified.
    """

    # The robot is spawned per env (instanceable=True allows many parallel envs).
    # usd_path is the CUSTOM_SCENE_USD placeholder; train_ur10e.py overrides it with a temp robot USD.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,  # "/World/envs/env_.*/ur10e"
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUSTOM_SCENE_USD,  # placeholder, overridden by train_ur10e.py
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Initial joint positions; remove to use
            # the defaults stored in the USD
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
            },
        ),
        actuators={
            # IsaacLab needs an ActuatorCfg to know how to send commands.
            # stiffness=None, damping=None -> use the USD DriveAPI values (not overridden)
            # effort_limit_sim=None -> use the USD effort limits
            # This keeps the USD values unchanged during training.
            # The actuator must cover all 6 arm joints
            # so wrist_2/wrist_3 can be held at their setpoints.
            "arm": ImplicitActuatorCfg(
                joint_names_expr=JOINT_NAMES_EXPR,
                effort_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_JOINT_NAMES_EXPR,
                effort_limit_sim=None,
                stiffness=None,
                damping=None,
            ),
            # 2F-140 distal finger DOFs (outer_knuckle / outer_finger / inner_finger /
            # inner_finger_pad). In the USD these are driven by a MIMIC constraint that is
            # NOT applied in sim → they are passive (no drive), so set_joint_position_target
            # was ignored and the pad never wrapped (pads met at the tip). Giving them an
            # explicit position drive keeps them at the open position (0) during training.
            "gripper_distal": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_DISTAL_JOINTS_EXPR,
                effort_limit_sim=None,
                stiffness=GRIPPER_DRIVE_STIFFNESS,
                damping=GRIPPER_DRIVE_DAMPING,
            ),
        },
    )

    # Extra light (remove if the USD already has enough lighting)
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

    # Object (per env), moved at reset by EventCfg.place_cylinder_under_gripper.
    # spawn = primitive orange cylinder by default, or the OBJECT_USD ref (cube/hexprism/plusprism).
    # Scene entity key stays "cylinder" so all obs/reward/placement code is unchanged.
    cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=_make_object_spawn(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )

    # Wrist camera (RGB + depth). TiledCamera uses one tiled render product for all env cameras,
    # which is much lighter on RTX descriptor/ParameterBlock resources than one render product per env.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        spawn=None,
        data_types=["rgb", "distance_to_image_plane"],
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )


