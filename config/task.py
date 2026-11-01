from __future__ import annotations

"""Tunable task definition: object geometry, GT target, tolerances, scripted grasp.

TUNABLE — edit values here. These describe WHAT the task is (object size, where
the target sits, success bands, the play-only scripted grasp/lift). The robot
wiring (joint name lists, prim paths) stays in the scene config; the control/
reward IMPLEMENTATIONS that consume these numbers are fixed library code.
"""

from .env_flags import env_float, env_str

# ── Object: silinder oranye, diameter 1.5 cm, tinggi 10 cm, massa 100 gram. ──
OBJECT_CYL_RADIUS_M = 0.0075   # 1.5 cm diameter -> 0.75 cm radius
OBJECT_CYL_MASS_KG = 0.10      # 100 gram
# Object HEIGHT + FLOOR-Z are env-overridable (for other objects): the driver
# sets OBJECT_CYL_HEIGHT_M per object so BOTH the reset placement (root = floor_z + 0.5*height → base
# on the ground) AND the GT target top (_gt_object_top_z_w = root_z + 0.5*height) track the object's
# own top. CYLINDER_FLOOR_Z_M seats the base on the floor when a USD's origin is not centered (floor_z
# = -0.5*height for base-origin). Primitive shapes made for other objects are
# authored centre-origin with a uniform 0.10m height, so they reuse the cylinder values (0.10/0.0).
# Unset → the current cylinder values, so variants 1-8 are UNCHANGED.
OBJECT_CYL_HEIGHT_M = env_float("OBJECT_CYL_HEIGHT_M", 0.10)     # 10 cm (cylinder default)
CYLINDER_FLOOR_Z_M = env_float("CYLINDER_FLOOR_Z_M", 0.000)      # cylinder on the floor (Z=0)
OBJECT_FLOOR_CLEARANCE_M = 0.002  # avoid initial floor penetration/contact jitter

# ── Env-overridable object USD (Level-2 YOLO dataset + per-object generalization) ──
# Unset/empty (default) → the primitive orange CylinderCfg above (variants 1-8 UNCHANGED).
# When set to a USD path (e.g. assets/usd/object_gen/cube.usd) the scene spawns THAT rigid
# object instead, so the wrist camera renders that object (for example another primitive shape)
# for the in-scene YOLO dataset generator and for per-object RL deploy. The placement event moves the
# object's root pose each reset regardless of shape, so no other wiring changes.
OBJECT_USD = env_str("OBJECT_USD", "")
OBJECT_USD_SCALE = env_float("OBJECT_USD_SCALE", 1.0)  # uniform scale for the referenced USD

# ── GT target (privileged critic + GT reward) ──
# Target XY is cylinder XY; target Z is cylinder top + clearance. Cylinder root is its center.
GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M = 0.185  # ground clearance
# Precision-picking band. Tightened 0.008 → 0.003 → 0.001 (XY) / 0.008 → 0.004 (Z) so
# success/stay/idle + the play-mode scripted-grasp trigger reward true ~1 mm centering
# (the goal), not just "graspable". This is safe ONLY because the steep fine XY exp term
# (EXP_FINE_XY_SCALE_M=0.008 in config/rewards_weights.py) supplies gradient into sub-mm to reach this
# band. Exploration and training budget are tuned independently in config/training.py.
# Physical grasp still works far looser (Ø15mm cyl, 140mm gripper) — precision, not feasibility.
# Env-overridable (GT_SUCCESS_XY_TOL_M / GT_SUCCESS_Z_TOL_M): an action-scale sweep
# reverts the "stay area" to the pre-tuning 8mm/8mm band; unset → the sub-mm defaults below.
GT_SUCCESS_XY_TOL_M = env_float("GT_SUCCESS_XY_TOL_M", 0.001)
GT_SUCCESS_Z_TOL_M = env_float("GT_SUCCESS_Z_TOL_M", 0.001)
GT_MIN_CLEARANCE_ABOVE_TOP_M = 0.002

# ── Vision: YOLO detection-confidence threshold for the no-detection penalty ──
YOLO_CONFIDENCE_THRESHOLD_PENALTY = 0.40

# ── Scripted grasp/lift phase (PLAY only) after stay stabil ──
SCRIPTED_STAY_STEPS_REQUIRED = 20      # control steps inside the target box before the gripper closes
SCRIPTED_STAY_Z_TOL_M = GT_SUCCESS_Z_TOL_M  # the play grasp uses the same Z band as validation
SCRIPTED_GRIPPER_CLOSE_STEPS = 30      # play only: ramp close gripper (faster: was 80)
SCRIPTED_GRIPPER_HOLD_STEPS = 15       # play only: hold the closed gripper before lifting
SCRIPTED_LIFT_HEIGHT_M = 0.30          # object lifted 30 cm for a successful lift
SCRIPTED_LIFT_HOLD_STEPS = 20          # object must stay above the lift target for N steps
SCRIPTED_LIFT_SUCCESS_TOL_M = 0.02     # toleransi sukses lift object
SCRIPTED_LIFT_EXTRA_HEIGHT_M = 0.20    # extra lift height so the EE does not stop before the object rises
SCRIPTED_LIFT_SPEED_M_S = 0.10         # IK lift speed (faster so the 30cm lift finishes before timeout)
SCRIPTED_IK_MAX_JOINT_STEP_RAD = 0.025 # per-step joint delta clamp during the IK lift (0.10 m/s lift is not throttled)
# 2F-140 finger_joint full close = 45° = 0.7854 rad (verified in Isaac Sim GUI; the
# joint angle there is degrees, IsaacLab joint_pos is radians). Commanding the max
# fully closes the fingers; with an object the fingers stop on it and the residual
# position error gives continuous grip force ("rapat"). If you swap back to a 2F-85,
# set this to that gripper's max instead.
SCRIPTED_GRIPPER_CLOSE_TARGETS = {
    "finger_joint": 0.7854,
}

# ── Weld-on-grasp (play scripted demo) ──
# The 2F-140 USD finger linkage is a hard constraint that only makes TIP contact, so the
# object is kinematically ATTACHED to the EE at close (rigid relative transform) and follows
# it through the lift. Pure play-only; does not touch RL training.
SCRIPTED_WELD_ON_GRASP = True

# ── 2F-140 underactuated multi-joint close ──
# The 2F-140 is UNDERACTUATED: driving only finger_joint rotates the outer knuckle so the
# fingertip swings in, but the distal PAD (inner_finger_joint) must COUNTER-rotate (mimic
# ≈ -1) to stay parallel and WRAP the object. The play scripted close drives EVERY finger DOF
# it can find, mapping each by a name SUBSTRING → multiplier. Run play once and read the
# one-time "[SCRIPTED][joints]" dump to confirm the real joint names/signs in YOUR USD, then
# tune. Set GRIPPER_DRIVE_UNDERACTUATED=False to fall back to finger_joint-only (USD mimic).
GRIPPER_DRIVE_UNDERACTUATED = True
# Distal finger DOFs (everything except the driver finger_joint). Mimic-driven in the USD;
# the mimic isn't applied in sim so they are PASSIVE and need an explicit position drive.
GRIPPER_DISTAL_JOINTS_EXPR = [
    ".*outer_knuckle_joint",
    ".*outer_finger_joint",
    ".*inner_finger_joint",
    ".*inner_finger_pad_joint",
]
# Position-drive gains for the distal joints. The USD has a physical 4-bar linkage pinning
# these as a function of the knuckle; 200 proved the distal linkage is a HARD constraint
# (25× stiffness barely moved inner_finger). finger_joint USD stiff ≈ 57000 for reference.
GRIPPER_DRIVE_STIFFNESS = 200.0
GRIPPER_DRIVE_DAMPING = 20.0
GRIPPER_FULL_CLOSE_RAD = 0.7854   # finger_joint full close (45°); the driver magnitude
# Keys matched MOST-SPECIFIC-FIRST (longest substring wins), driver by exact name.
# Multiplier = closed angle / GRIPPER_FULL_CLOSE_RAD. If the wrap looks wrong, first flip the
# inner_finger sign (-1 ↔ +1). Set a value to 0.0 to HOLD flat; remove a key to leave passive.
GRIPPER_CLOSE_MIMIC = {
    "finger_joint": 1.0,       # driver (left outer knuckle) — exact-name match
    "outer_knuckle": 1.0,      # right outer knuckle: mirror, closes inward (limit [0,1.309] ⇒ +)
    "inner_finger_pad": 0.0,   # PAD face: hold flat (do NOT tilt to the ±45° limit)
    "inner_finger": 1.0,       # parallelogram inner bar: TRACK the knuckle (+1) so the pad
                               #   (coupler) stays parallel. Flip to -1 if the pad splays OPEN.
    "outer_finger": 0.0,       # passive coupling link: hold at nominal to stop flopping
}

# ── Approach axis for crane/EE orientation checks (0=X,1=Y,2=Z). ──
APPROACH_AXIS = 2
CRANE_TARGET_AXIS_W = (0.0, 0.0, -1.0)

# ── Reset randomization: object xyz offset ranges (metres). ──
CARTESIAN_OBJECT_OFFSET_X_RANGE_M = (-0.04, 0.04)
CARTESIAN_OBJECT_OFFSET_Y_RANGE_M = (-0.04, 0.04)
# Keep Z fixed by default so the cylinder stays on the floor; non-zero values are supported.
CARTESIAN_OBJECT_OFFSET_Z_RANGE_M = (0.0, 0.0)
