"""Environment/body index helpers without control, Isaac or policy imports."""
from collections.abc import Sequence

import torch


def env_ids_tensor(env, env_ids: torch.Tensor | slice | Sequence[int] | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if not isinstance(env_ids, torch.Tensor):
        return torch.tensor(env_ids, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def just_reset(env) -> torch.Tensor:
    """[N] bool: the env is on the first control step of a new episode."""
    return env.episode_length_buf <= 1


def find_body_id(robot, body_name: str) -> int | None:
    """Articulation body index for an exact name, or None when absent."""
    body_ids, _ = robot.find_bodies([str(body_name)], preserve_order=True)
    return int(body_ids[0]) if len(body_ids) > 0 else None
