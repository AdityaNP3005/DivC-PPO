from __future__ import annotations

"""Tunable task definition: object geometry, GT target, tolerances, gripper drive.

TUNABLE — edit values here. These describe WHAT the task is (object size, where
the target sits, success bands). The robot
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
# success/stay/idle reward true ~1 mm centering
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

# Stable success: consecutive control steps inside the 1 mm target box (about 0.67 s at 30 Hz).
SCRIPTED_STAY_STEPS_REQUIRED = 20

# ── Robotiq 2F-140 finger joints ──
# The distal finger DOFs are mimic-driven in the USD, but the mimic is not applied in sim, so
# they are passive and get an explicit position drive that holds the open gripper.
GRIPPER_DISTAL_JOINTS_EXPR = [
    ".*outer_knuckle_joint",
    ".*outer_finger_joint",
    ".*inner_finger_joint",
    ".*inner_finger_pad_joint",
]
GRIPPER_DRIVE_STIFFNESS = 200.0
GRIPPER_DRIVE_DAMPING = 20.0

# ── Approach axis for crane/EE orientation checks (0=X,1=Y,2=Z). ──
APPROACH_AXIS = 2
CRANE_TARGET_AXIS_W = (0.0, 0.0, -1.0)

# ── Reset randomization: object xyz offset ranges (metres). ──
CARTESIAN_OBJECT_OFFSET_X_RANGE_M = (-0.04, 0.04)
CARTESIAN_OBJECT_OFFSET_Y_RANGE_M = (-0.04, 0.04)
# Keep Z fixed by default so the cylinder stays on the floor; non-zero values are supported.
CARTESIAN_OBJECT_OFFSET_Z_RANGE_M = (0.0, 0.0)
