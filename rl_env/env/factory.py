"""The one way training and validation build the Isaac env (after AppLauncher)."""

import gymnasium as gym
import torch

from rl_env.camera_utils import configure_camera_sensor
from rl_env.project_config import CUSTOM_SCENE_USD
from rl_env.usd_utils import patch_and_load_stage


def make_isaac_env(cfg, env_id, *, num_envs, seed=None, device=None, camera=None, render_mode=None):
    """Size the cfg, configure the wrist camera, load the patched USD stage and gym.make the env.

    camera: (resolution_text, update_period, label) or None to keep the cfg's sensor.
    device: "cuda:N" for cfg.sim.device and the current CUDA device, or None for the cfg default.
    Returns the unwrapped-gym env; callers add Sb3VecEnvWrapper (training probes the camera first).
    """
    cfg.scene.num_envs = int(num_envs)
    if seed is not None:
        cfg.seed = seed
    if device is not None:
        cfg.sim.device = device
    if camera is not None:
        configure_camera_sensor(cfg, *camera)
    if device is not None:
        try:
            torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        except Exception:
            pass
    patch_and_load_stage(CUSTOM_SCENE_USD, cfg)
    return gym.make(env_id, cfg=cfg, render_mode=render_mode)
