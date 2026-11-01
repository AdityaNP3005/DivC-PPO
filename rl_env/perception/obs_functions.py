from __future__ import annotations

"""Deployable observation terms: YOLO center/error obs (from the detection cache) and EE delta."""

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from config import (
    YOLO_CACHE_NORM_DEPTH_M,
    YOLO_CACHE_NORM_PX,
    YOLO_TRAIN_BBOX_CENTER_JITTER_STD,
)
from config.env_flags import env_bool
from rl_env.perception.yolo_cache import _get_yolo_detection_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def yolo_detection_obs(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    include_depth: bool = False,
) -> torch.Tensor:
    """Normalized YOLO center [cx, cy] (+ box-area depth when ``include_depth``)."""
    cache = _get_yolo_detection_cache(env, sensor_cfg)
    centers = _observed_centers(cache)
    if include_depth:
        return torch.cat([centers, cache["depth_area_norm"]], dim=-1)
    return centers


def yolo_detection_error_obs(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    target_center_px: tuple[float, float],
    target_depth_m: float,
    clip: float = 10.0,
    include_depth: bool = False,
) -> torch.Tensor:
    """Signed YOLO error vs the visual target, normalized like the cache.

    [dx, dy] (+ ddepth when ``include_depth``); dx, dy > 0 means the detection is
    right/down of the target.
    """
    cache = _get_yolo_detection_cache(env, sensor_cfg)
    target_center = torch.tensor(
        [float(target_center_px[0]) / YOLO_CACHE_NORM_PX, float(target_center_px[1]) / YOLO_CACHE_NORM_PX],
        device=env.device,
        dtype=torch.float32,
    )
    # Same jitter sample as yolo_detection_obs; detector cache, depth, rewards and GT stay clean.
    error = _observed_centers(cache) - target_center
    if include_depth:
        depth_error = cache["depth_area_norm"] - (float(target_depth_m) / YOLO_CACHE_NORM_DEPTH_M)
        error = torch.cat([error, depth_error], dim=-1)
    return torch.clamp(error, -float(clip), float(clip))


_JITTER_ANNOUNCED = False


@lru_cache(maxsize=1)
def _bbox_jitter_std() -> tuple[float, str]:
    """Bbox-center jitter std (normalized px) for this process, resolved once.

    The noise evaluation sets YOLO_OBS_NOISE_STD explicitly (zero keeps it clean); otherwise
    training (train_ur10e sets YOLO_TRAIN_BBOX_CENTER_JITTER_ACTIVE) uses the config std.
    """
    raw_eval = os.environ.get("YOLO_OBS_NOISE_STD")
    if raw_eval is not None and raw_eval.strip():
        try:
            return max(0.0, float(raw_eval)), "noise eval"
        except ValueError:
            return 0.0, "noise eval"
    if env_bool("YOLO_TRAIN_BBOX_CENTER_JITTER_ACTIVE", False):
        return float(YOLO_TRAIN_BBOX_CENTER_JITTER_STD), "training"
    return 0.0, "off"


def _observed_centers(cache: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cache centers, with one shared jitter sample per cache fill when jitter is on."""
    global _JITTER_ANNOUNCED
    std, source = _bbox_jitter_std()
    if std <= 0.0:
        return cache["center_norm"]
    if not _JITTER_ANNOUNCED:
        print(
            f"[YOLO-BBOX-JITTER] source={source} sigma={std:g} normalized "
            f"({std * YOLO_CACHE_NORM_PX:g}px @ {YOLO_CACHE_NORM_PX:g}px)"
        )
        _JITTER_ANNOUNCED = True
    if cache.get("center_obs_step") != cache["step"]:
        centers = cache["center_norm"]
        # Missing detections stay at (0, 0).
        jittered = centers + torch.randn_like(centers) * std * cache["has_det"].reshape(-1, 1)
        cache["center_obs_norm"] = torch.clamp(jittered, 0.0, 1.0)
        cache["center_obs_step"] = cache["step"]
    return cache["center_obs_norm"]


def ee_delta_pos_obs(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    clip: float = 10.0,
) -> torch.Tensor:
    """Current EE xyz position relative to the configured initial FK pose.

    The baseline is captured from the fixed initial joint setting, so the start
    pose is [0, 0, 0] and values are raw meters, not normalized.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore[index]

    origin = getattr(env, "_cartesian_initial_fk_ee_pos_w", None)
    if origin is None or tuple(origin.shape) != tuple(ee_pos_w.shape):
        origin = ee_pos_w.detach().clone()
        setattr(env, "_cartesian_initial_fk_ee_pos_w", origin)

    out = ee_pos_w - origin
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(out, -float(clip), float(clip))
