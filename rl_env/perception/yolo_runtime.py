from __future__ import annotations

"""YOLO runtime: SAM-free Ultralytics import, lazy model singleton, inference
settings, gripper exclusion, and robust depth extraction helpers."""

import importlib
import math
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


_YOLO_MODEL = None


def _prepare_ultralytics_runtime() -> None:
    """Prepare a YOLO-only Ultralytics runtime for Isaac Sim.

    Torchvision is intentionally cleared: both Isaac's prebundle and the current
    storage venv copy can mismatch the active Torch build. Ultralytics detect
    inference can use its pure TorchNMS path as long as torchvision is not loaded.

    Keep sys.path order untouched. Isaac Sim, SB3, and Torch are sensitive to
    mixing packages from Isaac's Python tree and a separate package tree after startup.
    """
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)


def _patch_ultralytics_network_checks() -> None:
    """Disable Ultralytics GitHub asset checks for offline/restricted environments."""
    try:
        from ultralytics.utils import downloads as ultralytics_downloads
        try:
            from ultralytics.utils import SETTINGS as ultralytics_settings
        except Exception:
            ultralytics_settings = None

        ultralytics_downloads.get_github_assets = lambda *args, **kwargs: ("", [])
        ultralytics_downloads.attempt_download_asset = lambda file, *args, **kwargs: str(file)
        if ultralytics_settings is not None:
            try:
                ultralytics_settings.update({"sync": False})
            except Exception:
                pass
    except Exception:
        # Do not interrupt training/validation if the patch is unavailable.
        pass


def _import_yolo_class_without_sam():
    """Import Ultralytics YOLO without importing SAM/torchvision.

    Ultralytics 8.4 lazy `from ultralytics import YOLO` imports
    `ultralytics.models.__init__`, which imports SAM and then torchvision.
    Isaac Sim often ships a mismatched torchvision, so for this detection-only
    pipeline we load only `ultralytics.models.yolo.model.YOLO`.
    """
    _prepare_ultralytics_runtime()
    import ultralytics

    models_dir = Path(ultralytics.__file__).resolve().parent / "models"
    for name in list(sys.modules):
        if name == "ultralytics.models" or name.startswith("ultralytics.models."):
            sys.modules.pop(name, None)

    models_pkg = types.ModuleType("ultralytics.models")
    models_pkg.__path__ = [str(models_dir)]
    models_pkg.__package__ = "ultralytics.models"
    models_pkg.__file__ = str(models_dir / "__init__.py")
    sys.modules["ultralytics.models"] = models_pkg
    setattr(ultralytics, "models", models_pkg)

    return importlib.import_module("ultralytics.models.yolo.model").YOLO


def _get_yolo_model():
    global _YOLO_MODEL
    _prepare_ultralytics_runtime()
    if _YOLO_MODEL is None:
        _patch_ultralytics_network_checks()
        YOLO = _import_yolo_class_without_sam()

        from config import YOLO_DEFAULTS
        from evaluation.checkpoint_assets import require_detector

        # Entrypoints publish YOLO_MODEL_PATH (per-object detector); a missing file is a hard
        # error, never a silent fallback to a generic detector.
        model_path = require_detector(os.environ.get("YOLO_MODEL_PATH") or YOLO_DEFAULTS["model_path"])
        print(f"[YOLO] loading weights from: {model_path}")
        try:
            _YOLO_MODEL = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}") from e
    return _YOLO_MODEL


def _get_yolo_settings() -> tuple[float, float, str, int]:
    conf = float(os.environ.get("YOLO_CONF", "0.25"))
    iou = float(os.environ.get("YOLO_IOU", "0.45"))
    device = os.environ.get("YOLO_DEVICE", "cpu")
    max_det = int(os.environ.get("YOLO_MAX_DET", "1"))
    return conf, iou, device, max_det


def _scaled_gripper_exclusion_polygons_px(
    image_width: int = 640,
    image_height: int = 640,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    from config import (
        YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX,
        YOLO_GRIPPER_EXCLUSION_POLYGONS_PX,
    )

    canonical_w, canonical_h = YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX
    scale_x = float(image_width) / float(canonical_w)
    scale_y = float(image_height) / float(canonical_h)
    return tuple(
        tuple((float(x) * scale_x, float(y) * scale_y) for x, y in polygon)
        for polygon in YOLO_GRIPPER_EXCLUSION_POLYGONS_PX
    )


def _point_in_polygon_inclusive(
    px: float,
    py: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-7
            and min(x1, x2) - 1e-7 <= px <= max(x1, x2) + 1e-7
            and min(y1, y2) - 1e-7 <= py <= max(y1, y2) + 1e-7
        ):
            return True
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


def _center_outside_gripper_exclusion(
    cx: float,
    cy: float,
    image_width: int = 640,
    image_height: int = 640,
) -> bool:
    from config import YOLO_ROI_ENABLED

    if not YOLO_ROI_ENABLED:
        return True
    return not any(
        _point_in_polygon_inclusive(cx, cy, polygon)
        for polygon in _scaled_gripper_exclusion_polygons_px(image_width, image_height)
    )


def _select_best_unmasked_detection(
    xyxy,
    confs,
    image_width: int = 640,
    image_height: int = 640,
) -> int:
    best_index = -1
    best_confidence = -1.0
    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(value) for value in box]
        confidence = float(confs[index])
        if confidence <= best_confidence:
            continue
        if not _center_outside_gripper_exclusion(
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            image_width=image_width,
            image_height=image_height,
        ):
            continue
        best_index = index
        best_confidence = confidence
    return best_index


@lru_cache(maxsize=8)
def _gripper_exclusion_mask(image_height: int, image_width: int) -> np.ndarray:
    from config import YOLO_ROI_ENABLED

    mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
    if not YOLO_ROI_ENABLED:
        return mask.astype(bool)
    try:
        import cv2
    except Exception:
        return mask.astype(bool)
    polygons = [
        np.asarray(polygon, dtype=np.int32)
        for polygon in _scaled_gripper_exclusion_polygons_px(image_width, image_height)
    ]
    cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool)


def _draw_gripper_exclusion(img, color=(255, 0, 0), alpha: float = 0.28, thickness: int = 2):
    from config import YOLO_ROI_ENABLED

    if not YOLO_ROI_ENABLED or img is None:
        return img
    try:
        import cv2
    except Exception:
        return img
    height, width = img.shape[:2]
    polygons = [
        np.asarray(polygon, dtype=np.int32)
        for polygon in _scaled_gripper_exclusion_polygons_px(width, height)
    ]
    overlay = img.copy()
    cv2.fillPoly(overlay, polygons, color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
    cv2.polylines(img, polygons, True, color, thickness)
    cv2.putText(
        img,
        "GRIPPER EXCLUSION",
        (10, max(24, height - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return img


def _depth_2d(depth_img: torch.Tensor) -> torch.Tensor:
    """[H, W] depth from an [H, W] or [H, W, 1] camera output."""
    return depth_img[..., 0] if depth_img.dim() == 3 else depth_img


def _extract_depth_at_pixel(depth_img: torch.Tensor, px: int, py: int) -> float:
    depth_img = _depth_2d(depth_img)
    h, w = depth_img.shape[:2]
    px = max(0, min(w - 1, int(px)))
    py = max(0, min(h - 1, int(py)))
    if _gripper_exclusion_mask(h, w)[py, px]:
        return 0.0
    depth_val = depth_img[py, px].item()
    if depth_val == float("inf") or math.isnan(depth_val):
        return 0.0
    return float(depth_val)


def _extract_area_depth_in_detection_box(depth_img: torch.Tensor, x1: int, y1: int, x2: int, y2: int) -> float:
    """Extract robust object depth from valid pixels inside the detection area.

    We use a low percentile instead of a single center pixel or raw min. This
    still represents the object surface visible in the detection box, but is less noisy
    than taking one closest pixel.
    """
    depth_img = _depth_2d(depth_img)
    h, w = depth_img.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x1 >= x2 or y1 >= y2:
        return 0.0

    region = depth_img[y1:y2 + 1, x1:x2 + 1]
    excluded = torch.from_numpy(_gripper_exclusion_mask(h, w)[y1:y2 + 1, x1:x2 + 1])
    valid = region[~torch.isinf(region) & ~torch.isnan(region) & ~excluded]
    valid = valid[valid > 0.0]
    if valid.numel() == 0:
        return 0.0

    from config import YOLO_DEPTH_PERCENTILE

    depth_val = torch.quantile(valid.float(), float(YOLO_DEPTH_PERCENTILE)).item()
    return float(depth_val) if math.isfinite(float(depth_val)) else 0.0
