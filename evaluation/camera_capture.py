"""Draw the cached YOLO detection on a camera frame (play overlay, training probe).

The detection itself always comes from rl_env.perception.yolo_cache (the same values
the policy sees); this module only denormalizes and draws it. No Isaac Sim imports.

cv2 is optional: when unavailable, `_draw_detection` falls back to manual pixel
rectangles, exactly as the original did.
"""

import numpy as np
from rl_env.camera_utils import capture_rgb_uint8 as _to_uint8_rgb

try:
    import cv2
    _cv2_ok = True
except Exception:
    cv2 = None
    _cv2_ok = False


def _draw_detection(
    raw_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float,
    center_px: tuple[int, int],
    size_px: tuple[int, int],
    depth_m: float | None,
) -> np.ndarray:
    det_rgb = raw_rgb.copy()
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(det_rgb.shape[1] - 1, x2), min(det_rgb.shape[0] - 1, y2)
    if x2c <= x1c or y2c <= y1c:
        return det_rgb
    if _cv2_ok:
        color = (0, 220, 0)
        cv2.rectangle(det_rgb, (x1c, y1c), (x2c, y2c), color, 2)
        depth_text = f"Depth={depth_m:.3f}m" if depth_m is not None else "Depth=N/A"
        lines = [
            f"Conf={confidence:.2f}",
            f"Center=({center_px[0]},{center_px[1]})",
            f"Size=({size_px[0]},{size_px[1]})",
            depth_text,
        ]
        line_h = 18
        if y2c + line_h * len(lines) + 4 < det_rgb.shape[0]:
            text_x = x1c
            text_y = y2c + line_h
        else:
            text_x = x1c
            text_y = max(line_h, y1c - line_h * len(lines) - 4)
        for idx, line in enumerate(lines):
            y = min(det_rgb.shape[0] - 6, text_y + idx * line_h)
            cv2.putText(det_rgb, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        det_rgb[y1c:y1c+2, x1c:x2c, :] = (0, 255, 0)
        det_rgb[y2c-2:y2c, x1c:x2c, :] = (0, 255, 0)
        det_rgb[y1c:y2c, x1c:x1c+2, :] = (0, 255, 0)
        det_rgb[y1c:y2c, x2c-2:x2c, :] = (0, 255, 0)
    return det_rgb


def draw_cached_detection(raw_rgb: np.ndarray, cache: dict, index: int = 0) -> np.ndarray:
    """Box + labels for env ``index`` of the YOLO cache, always with the gripper exclusion."""
    from config import YOLO_CACHE_NORM_DEPTH_M, YOLO_CACHE_NORM_PX
    from rl_env.perception.yolo_runtime import _draw_gripper_exclusion

    det_rgb = raw_rgb.copy()
    if float(cache["has_det"][index]) > 0.5:
        cx, cy = (cache["center_norm"][index] * YOLO_CACHE_NORM_PX).tolist()
        w, h = (cache["size_norm"][index] * YOLO_CACHE_NORM_PX).tolist()
        depth_m = float(cache["depth_norm"][index, 0]) * YOLO_CACHE_NORM_DEPTH_M
        det_rgb = _draw_detection(
            raw_rgb,
            int(round(cx - 0.5 * w)), int(round(cy - 0.5 * h)),
            int(round(cx + 0.5 * w)), int(round(cy + 0.5 * h)),
            float(cache["conf"][index]),
            (int(round(cx)), int(round(cy))),
            (int(round(w)), int(round(h))),
            depth_m if depth_m > 0.0 else None,
        )
    return _draw_gripper_exclusion(det_rgb)
