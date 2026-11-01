from __future__ import annotations

"""Tunable perception knobs: camera, YOLO, cache normalization, and visual target.

TUNABLE — edit values here. No fixed-library code lives in this module; the YOLO
runtime/cache implementation reads these constants from `config`.
"""

import os

from config.env_flags import env_bool, env_float
from rl_env.project_config import ASSETS_ROOT

# Wrist camera prim already in the USD (pose untouched); YOLO is calibrated at this size.
CAMERA_PRIM_PATH = "/World/envs/env_.*/ur10e/ur10e/wrist_3_link/rsd455/RSD455/Camera_OmniVision_OV9782_Color"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640

CAMERA_DEFAULTS = {
    "resolution": f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
    "update_period": 0.0,
    "output_dir": None,
}

# `device="auto"` means: follow the selected process GPU. When --gpu 1 masks the
# process to physical GPU 1, YOLO should use local cuda:0 inside that process.
# Deployable weights are project-local; never fall back to another research run.
YOLO_DEFAULTS = {
    "model_path": f"{ASSETS_ROOT}/yolo/weights/best.pt",
    "conf": 0.25,
    "iou": 0.45,
    "device": "auto",
    "max_det": 1,
}



def publish_yolo_env(args) -> None:
    """Export the resolved detector settings for the YOLO cache (train, play, validator)."""
    os.environ["YOLO_MODEL_PATH"] = str(args.yolo_model_path)
    os.environ["YOLO_CONF"] = str(args.yolo_conf)
    os.environ["YOLO_IOU"] = str(args.yolo_iou)
    os.environ["YOLO_DEVICE"] = str(args.yolo_device)
    os.environ["YOLO_MAX_DET"] = str(args.yolo_max_det)

# Single source of truth for YOLO cache normalization.
# Must match CAMERA_WIDTH above and the max expected EE-to-target depth.
# The YOLO cache writes using these; train and play read them back.
YOLO_CACHE_NORM_PX: float = 640.0   # images are 640 px wide
YOLO_CACHE_NORM_DEPTH_M: float = 0.5  # max EE-to-cylinder depth ≈ 0.50 m

# Training-only bbox-center augmentation, in normalized-pixel units. For a
# 640px frame, sigma=0.01 is 6.4px. The actor sees the same sampled center in
# both its raw detection and signed-error terms; clean detector cache/rewards
# remain untouched. Values <= 0 disable the augmentation.
YOLO_TRAIN_BBOX_CENTER_JITTER_STD: float = max(
    0.0,
    env_float("YOLO_TRAIN_BBOX_CENTER_JITTER_STD", 0.0),
)

# Robust depth: percentile taken inside the detection bbox (10th = near surface).
YOLO_DEPTH_PERCENTILE: float = 0.10

# ── Visual target (SINGLE SOURCE) ──
# Used by BOTH the deployable YOLO-error observation AND the vision reward
# (variants 1 & 2). Defined here so the two can never drift apart — edit the
# depth/center ONCE here.
BBOX_TARGET_CENTER_PX: tuple[float, float] = (320.0, 400.0)
BBOX_TARGET_DEPTH_M: float = 0.19

# ── Full-frame gripper exclusion (SINGLE SOURCE) ──
# Detection stays active across the whole image except static polygons covering the
# open gripper's two jaws and lower dome/base. A bbox is rejected only when its CENTER
# lies inside a polygon. Polygon coordinates are canonical at 640x640 and scale with
# the live image size. Set YOLO_ROI_ENABLED=0 to bypass the filter.
YOLO_ROI_ENABLED: bool = env_bool("YOLO_ROI_ENABLED", True)
YOLO_GRIPPER_EXCLUSION_MODE: str = "full_frame_gripper_exclusion"
YOLO_GRIPPER_EXCLUSION_MASK_VERSION: int = 1
YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX: tuple[float, float] = (640.0, 640.0)

_YOLO_GRIPPER_LEFT_POLYGON_PX: tuple[tuple[float, float], ...] = (
    (0.0, 410.0),
    (90.0, 390.0),
    (130.0, 410.0),
    (220.0, 380.0),
    (220.0, 440.0),
    (265.0, 485.0),
    (195.0, 535.0),
    (95.0, 640.0),
    (0.0, 640.0),
)
YOLO_GRIPPER_EXCLUSION_POLYGONS_PX: tuple[tuple[tuple[float, float], ...], ...] = (
    _YOLO_GRIPPER_LEFT_POLYGON_PX,
    tuple((640.0 - x, y) for x, y in _YOLO_GRIPPER_LEFT_POLYGON_PX),
    (
        (75.0, 640.0),
        (115.0, 575.0),
        (165.0, 530.0),
        (255.0, 480.0),
        (385.0, 480.0),
        (475.0, 530.0),
        (525.0, 575.0),
        (565.0, 640.0),
    ),
)

# ── Vision reward (pure-vision arms) ──
# The reward error is YOLO vs BBOX_TARGET_* in METRIC-EQUIVALENT units, so the SAME
# reward terms/scales/tolerances apply as with GT (only the source differs).
# normalized-px error → meters-equivalent. 10 px / 640 ≈ 0.0156 ≈ GT XY tol 0.015 m, so gain 1.0
# keeps the XY tolerance/scale behaviour aligned with the GT terms.
VISION_REWARD_XY_GAIN_M: float = 1.0
# error magnitude (m-equiv) assigned to all axes when detection is lost → treated as "far".
VISION_REWARD_NO_DET_ERROR_M: float = 0.5
