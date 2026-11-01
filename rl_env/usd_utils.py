"""Shared USD patching utilities for UR10e training and evaluation.

Imported by train_ur10e.py and play.py, so a change here
applies to both.

Fungsi utama:
    patch_and_load_stage(src_usd, env_cfg)
        -> write the robot USD (ArticulationRoot fix) and the env USD (without the robot)
        → patch env_cfg.scene.robot.spawn.usd_path
        -> load the env USD as a sublayer of the active stage
        → return (robot_usd_path, env_usd_path)
"""

import os
import shutil
import tempfile


def _ensure_tmp_object_usd(src_usd: str) -> None:
    """Keep the stage's relative `object.usd` payload resolvable next to its temp copies.

    The source stage references `object.usd` beside `src_usd`; the temp stages live in
    tempfile.gettempdir() (which follows TMPDIR), so the payload is copied there.
    """
    src_dir = os.path.dirname(os.path.abspath(src_usd))
    src_object = os.path.join(src_dir, "object.usd")
    tmp_object = os.path.join(tempfile.gettempdir(), "object.usd")
    if os.path.exists(src_object):
        try:
            shutil.copy2(src_object, tmp_object)
            print(f"[INFO] [usd_utils] Copied dependency: {src_object} -> {tmp_object}")
        except Exception as e:
            print(f"[WARNING] [usd_utils] Gagal copy dependency object.usd: {e}")


def _temp_stage_copy(src: str, prefix: str):
    """Copy the source USD to a fresh temp file and open it (the original is never edited)."""
    from pxr import Usd

    fd, tmp = tempfile.mkstemp(suffix=".usd", prefix=prefix)
    os.close(fd)
    shutil.copy2(src, tmp)
    return tmp, Usd.Stage.Open(tmp)


def make_robot_usd(src: str) -> str:
    """Write a temp USD containing only /World/ur10e with the ArticulationRoot fix.

    Fixes applied:
        Nested ArticulationRoot: removed from /World/ur10e (Xform),
        kept only on root_joint (as PhysX requires).

    The source file is not modified. Returns the temp USD path.
    """
    from pxr import UsdPhysics, Sdf

    tmp, stage = _temp_stage_copy(src, "rl_robot_")

    ur10e = stage.GetPrimAtPath("/World/ur10e")
    if not ur10e.IsValid():
        print("[WARNING] [usd_utils] /World/ur10e not found in the USD!")
        return src

    # Fix: nested ArticulationRoot, removed from /World/ur10e and kept on root_joint
    rj = stage.GetPrimAtPath("/World/ur10e/root_joint")
    if (ur10e.HasAPI(UsdPhysics.ArticulationRootAPI) and
            rj.IsValid() and rj.HasAPI(UsdPhysics.ArticulationRootAPI)):
        spec = stage.GetRootLayer().GetPrimAtPath("/World/ur10e")
        if spec is not None:
            def _strip(items):
                return [s for s in (items or []) if "ArticulationRootAPI" not in s]
            cur = spec.GetInfo("apiSchemas")
            new_op = Sdf.TokenListOp()
            new_op.prependedItems = _strip(cur.prependedItems)
            new_op.appendedItems  = _strip(cur.appendedItems)
            new_op.deletedItems   = list(cur.deletedItems or [])
            if cur.isExplicit:
                new_op.explicitItems = _strip(cur.explicitItems)
            spec.SetInfo("apiSchemas", new_op)
            print("[INFO] [usd_utils] Fix: nested ArticulationRoot diperbaiki")

    # Fix: XformStack reset for the RigidBody on the camera mount (rsd455/RSD455)
    # Fix: nested RigidBody hierarchy error on the camera mount (rsd455/RSD455)
    # Error: "Rigid Body missing xformstack reset when child of another enabled rigid body"
    # Remove RigidBodyAPI from RSD455 (camera prim) so it does not conflict with the parent wrist_3_link
    # The camera only needs an Xform, not RigidBodyAPI
    rsd455_prims = [
        "/World/ur10e/wrist_3_link/rsd455/RSD455",
        "/World/ur10e/wrist_3_link/rsd455",
        "/World/ur10e/ur10e/wrist_3_link/rsd455/RSD455",
        "/World/ur10e/ur10e/wrist_3_link/rsd455",
    ]
    
    for prim_path in rsd455_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # Author a delete opinion even when the API comes from a referenced sensor USD.
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Nested camera rigid-body API remains: {prim_path}")
            print(f"[INFO] [usd_utils] Fix: PhysicsRigidBodyAPI removed from {prim_path}")

    # Remove every non-robot prim from /World (ground plane, PhysicsScene, etc.)
    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for child in list(world.GetChildren()):
            if child.GetName() != "ur10e":
                stage.RemovePrim(child.GetPath())

    stage.Save()
    print(f"[INFO] [usd_utils] Robot USD: {tmp}")
    return tmp


def make_env_usd(src: str) -> str:
    """Write a temp USD with the environment only (without /World/ur10e).

    IsaacLab spawns the robot per env from the robot USD.
    The source file is not modified. Returns the temp USD path.
    """
    tmp, stage = _temp_stage_copy(src, "rl_env_")

    if stage.GetPrimAtPath("/World/ur10e").IsValid():
        stage.RemovePrim("/World/ur10e")
        print("[INFO] [usd_utils] /World/ur10e removed from the env USD")

    stage.Save()
    print(f"[INFO] [usd_utils] Env USD: {tmp}")
    return tmp


def _select_unique_prim_name(stage, candidates, *, require_rigid_body: bool = False, require_joint: bool = False):
    """Select the first candidate that exists uniquely in the patched robot USD."""
    from pxr import UsdPhysics

    skipped = []
    for candidate in candidates:
        matches = []
        for prim in stage.Traverse():
            if prim.GetName() != candidate:
                continue
            if require_rigid_body and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if require_joint and "Joint" not in str(prim.GetTypeName()):
                continue
            matches.append(str(prim.GetPath()))

        if len(matches) == 1:
            return candidate, matches[0]
        if len(matches) > 1:
            skipped.append((candidate, matches))

    for candidate, matches in skipped:
        print(
            f"[WARNING] [usd_utils] Fallback candidate '{candidate}' is not unique, "
            f"skipped: {matches}"
        )
    return None, None


def _patch_scene_entity_body(params, key, ee_name):
    cfg = params.get(key) if params is not None else None
    if cfg is None or not hasattr(cfg, "body_names"):
        return
    # Only patch configs that already target a body. SceneEntityCfg("robot")
    # must stay as the whole articulation for scripted IK/gripper control.
    if getattr(cfg, "body_names", None) is not None:
        cfg.body_names = [ee_name]


def _patch_gripper_fallbacks(robot_usd: str, env_cfg) -> None:
    """Patch train/play cfg to the active gripper EE body.

    The controlled gripper joint stays fixed as finger_joint for both Robotiq
    2F-85 and 2F-140. Only the EE body name is allowed to fallback because USD
    exports may expose the gripper base as base_link_0 or base_link.
    """
    from pxr import Usd as _Usd

    from rl_env.env import EE_LINK_NAME_CANDIDATES

    stage = _Usd.Stage.Open(robot_usd)
    if stage is None:
        print(f"[WARNING] [usd_utils] Robot USD could not be opened for the fallback: {robot_usd}")
        return

    ee_name, ee_path = _select_unique_prim_name(
        stage, EE_LINK_NAME_CANDIDATES, require_rigid_body=True
    )

    if ee_name is None:
        ee_name = EE_LINK_NAME_CANDIDATES[0]
        print(
            f"[WARNING] [usd_utils] EE fallback found no unique candidate; "
            f"keeping '{ee_name}'."
        )
    else:
        print(f"[INFO] [usd_utils] EE body aktif: {ee_name} ({ee_path})")

    # Cartesian action target body.
    arm_action = getattr(getattr(env_cfg, "actions", None), "arm_action", None)
    if arm_action is not None and hasattr(arm_action, "body_name"):
        arm_action.body_name = ee_name

    # Every observation/reward/event term whose entity already targets a body.
    # Gripper joints stay finger_joint; whole-articulation entities stay untouched.
    groups = (
        getattr(getattr(env_cfg, "observations", None), "policy", None),
        getattr(env_cfg, "rewards", None),
        getattr(env_cfg, "events", None),
    )
    for group in groups:
        for term_name in dir(group) if group is not None else ():
            if term_name.startswith("_"):
                continue
            params = getattr(getattr(group, term_name, None), "params", None)
            if not isinstance(params, dict):
                continue
            for key in ("ee_cfg", "asset_cfg", "robot_cfg"):
                _patch_scene_entity_body(params, key, ee_name)


def patch_and_load_stage(src_usd: str, env_cfg) -> tuple:
    """Main entry point: patch env_cfg and load the env sublayer into the stage.

    Langkah:
        1. Write robot_usd (ArticulationRoot fix, only /World/ur10e)
        2. Write env_usd (the whole scene without the robot)
        3. Patch env_cfg.scene.robot.spawn.usd_path → robot_usd
        4. Load env_usd as a sublayer of the active Omniverse stage

    Args:
        src_usd  : source USD path (not modified)
        env_cfg  : ManagerBasedRLEnvCfg instance to patch

    Returns:
        (robot_usd_path, env_usd_path), the temp USD paths written
    """
    import omni.usd as _omni_usd

    _ensure_tmp_object_usd(src_usd)
    robot_usd = make_robot_usd(src_usd)
    env_usd   = make_env_usd(src_usd)

    _patch_gripper_fallbacks(robot_usd, env_cfg)

    # Patch env_cfg so the robot spawns from the temp robot USD
    env_cfg.scene.robot.spawn.usd_path = robot_usd

    # Load the env USD as a sublayer of the active stage
    stage = _omni_usd.get_context().get_stage()
    if stage is not None:
        if env_usd not in list(stage.GetRootLayer().subLayerPaths):
            stage.GetRootLayer().subLayerPaths.append(env_usd)
        print(f"[INFO] [usd_utils] Env USD loaded as a sublayer: {env_usd}")
    else:
        print("[WARNING] [usd_utils] Stage not available; env USD not loaded!")

    return robot_usd, env_usd
