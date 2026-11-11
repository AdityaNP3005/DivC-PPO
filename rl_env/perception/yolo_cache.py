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
    _best_unmasked_detections,
    _get_yolo_model,
    _get_yolo_settings,
    _tensor_box_predictor,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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
        rgb = sensor.data.output[rgb_type][..., :3]
        depth_maps = sensor.data.output[depth_type]  # stays on the GPU: only the detection boxes are read
    except Exception as e:
        raise RuntimeError(
            f"Failed to read camera outputs '{rgb_type}'/'{depth_type}' from sensor '{sensor_cfg.name}': {e}"
        ) from e

    conf, iou, device, max_det = _get_yolo_settings()
    # Isaac camera tensors are RGB on the GPU and Ultralytics takes an RGB BCHW 0-1 tensor as is (the square
    # frame needs no letterbox), so no image is copied to the CPU. Extra candidates let the exclusion filter
    # recover the object after the gripper box.
    images = rgb.permute(0, 3, 1, 2).float().div_(255.0).contiguous()
    try:
        model = _get_yolo_model()
        model.predict(
            images, conf=conf, iou=iou, device=device, verbose=False, max_det=max(int(max_det), 5),
            predictor=_tensor_box_predictor(),
        )
        # NMS, the gripper filter and the box depths run on the whole batch on the GPU.
        det = _best_unmasked_detections(*model.predictor.top, depth_maps[..., 0])
    except Exception as e:
        raise RuntimeError(f"YOLO detection pipeline failed at step {step}: {e}") from e

    x1, y1, x2, y2 = det["xyxy"].unbind(1)
    cache["center_norm"] = torch.stack([x1 + x2, y1 + y2], dim=-1) * 0.5 / YOLO_CACHE_NORM_PX
    cache["size_norm"] = torch.stack([x2 - x1, y2 - y1], dim=-1).clamp(min=0.0) / YOLO_CACHE_NORM_PX
    cache["depth_norm"] = det["depth"].unsqueeze(-1) / YOLO_CACHE_NORM_DEPTH_M
    cache["depth_area_norm"] = det["depth_area"].unsqueeze(-1) / YOLO_CACHE_NORM_DEPTH_M
    cache["conf"] = det["conf"]
    cache["has_det"] = det["has_det"]
    cache["step"] = step
    return cache
