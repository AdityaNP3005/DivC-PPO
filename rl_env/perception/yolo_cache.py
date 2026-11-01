from __future__ import annotations

"""YOLO detection cache: the only place the wrist camera is run through YOLO.

One inference per control-step scene state (reset invalidates it), stored on
env._yolo_detection_cache and read by observations, vision rewards, the training
probe and monitor. Values are normalized to [0, 1]:
    center_norm [N,2], size_norm [N,2]   pixels / YOLO_CACHE_NORM_PX
    depth_norm [N,1]                     box-area depth, else center depth, / YOLO_CACHE_NORM_DEPTH_M
    depth_area_norm [N,1]                box-area depth only (0 when unavailable)
    conf [N], has_det [N] (1.0/0.0), step
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from config import YOLO_CACHE_NORM_DEPTH_M, YOLO_CACHE_NORM_PX
from rl_env.perception.yolo_runtime import (
    _extract_area_depth_in_detection_box,
    _extract_depth_at_pixel,
    _get_yolo_model,
    _get_yolo_settings,
    _select_best_unmasked_detection,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# cx, cy, w, h, depth, depth_area, conf, has_det
_NO_DETECTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _best_detection_row(result, depth_img: torch.Tensor, image_width: int, image_height: int) -> tuple:
    if result.boxes is None or len(result.boxes) == 0:
        return _NO_DETECTION
    confs = result.boxes.conf.detach().cpu().numpy()
    xyxy_all = result.boxes.xyxy.detach().cpu().numpy()
    best_idx = _select_best_unmasked_detection(
        xyxy_all, confs, image_width=image_width, image_height=image_height
    )
    if best_idx < 0:  # every candidate center was inside the gripper exclusion
        return _NO_DETECTION
    x1, y1, x2, y2 = xyxy_all[best_idx].tolist()
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    # Task depth uses the detection area (gripper pixels excluded); center pixel is the fallback.
    depth_area = _extract_area_depth_in_detection_box(depth_img, x1, y1, x2, y2)
    depth = depth_area if depth_area > 0.0 else _extract_depth_at_pixel(depth_img, int(round(cx)), int(round(cy)))
    return (cx, cy, max(0.0, x2 - x1), max(0.0, y2 - y1), depth, depth_area, float(confs[best_idx]), 1.0)


def _get_yolo_detection_cache(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    rgb_type: str = "rgb",
    depth_type: str = "distance_to_image_plane",
) -> dict[str, torch.Tensor]:
    cache = getattr(env, "_yolo_detection_cache", None)
    if cache is None:
        cache = {}
        env._yolo_detection_cache = cache
    step = getattr(env, "common_step_counter", -1)
    if cache.get("step") == step:
        return cache

    try:
        sensor = env.scene.sensors[sensor_cfg.name]
    except KeyError as e:
        raise KeyError(f"Wrist camera sensor '{sensor_cfg.name}' not found in env.scene.sensors") from e
    try:
        rgb_np = sensor.data.output[rgb_type].detach().cpu().numpy()
        depth_cpu = sensor.data.output[depth_type].detach().cpu()
    except Exception as e:
        raise RuntimeError(
            f"Failed to read camera outputs '{rgb_type}'/'{depth_type}' from sensor '{sensor_cfg.name}': {e}"
        ) from e

    conf, iou, device, max_det = _get_yolo_settings()
    # Isaac camera tensors are RGB; Ultralytics numpy inputs are BGR. Request extra
    # candidates so the exclusion filter can recover the object after the gripper box.
    images = [img[..., :3][..., ::-1].copy() for img in rgb_np]
    try:
        results = _get_yolo_model().predict(
            images, conf=conf, iou=iou, device=device, verbose=False, max_det=max(int(max_det), 5)
        )
        rows = [
            _best_detection_row(result, depth_cpu[i], rgb_np[i].shape[1], rgb_np[i].shape[0])
            for i, result in enumerate(results)
        ]
    except Exception as e:
        raise RuntimeError(f"YOLO detection pipeline failed at step {step}: {e}") from e

    cx, cy, w, h, depth, depth_area, conf_t, has_det = (
        torch.tensor(column, device=env.device, dtype=torch.float32) for column in zip(*rows)
    )
    cache["center_norm"] = torch.stack([cx, cy], dim=-1) / YOLO_CACHE_NORM_PX
    cache["size_norm"] = torch.stack([w, h], dim=-1) / YOLO_CACHE_NORM_PX
    cache["depth_norm"] = depth.unsqueeze(-1) / YOLO_CACHE_NORM_DEPTH_M
    cache["depth_area_norm"] = depth_area.unsqueeze(-1) / YOLO_CACHE_NORM_DEPTH_M
    cache["conf"] = conf_t
    cache["has_det"] = has_det
    cache["step"] = step
    return cache
