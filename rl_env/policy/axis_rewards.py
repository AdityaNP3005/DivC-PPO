from __future__ import annotations

"""Map IsaacLab's weighted reward terms into deployable Cartesian XYZ streams."""

import math
from collections.abc import Sequence

import torch

from config.reward_units import validate_reward_units
from rl_env.env_wrappers import find_isaac_env


AXIS_REWARD_TERMS = (
    frozenset(("penalty_x", "abs_x", "exp_x_coarse", "exp_x", "gt_idle_x", "action_rate_x")),
    frozenset(("penalty_y", "abs_y", "exp_y_coarse", "exp_y", "gt_idle_y", "action_rate_y")),
    frozenset(("penalty_z", "abs_z", "exp_z_coarse", "exp_z", "gt_idle_z", "action_rate_z", "low_depth")),
)
SHARED_REWARD_TERMS = frozenset(
    ("success_once", "stay_xy", "alive_penalty", "no_detection_penalty", "joint_limit_clamp")
)
REQUIRED_REWARD_TERMS = frozenset().union(*(terms - {"low_depth"} for terms in AXIS_REWARD_TERMS))
KNOWN_REWARD_TERMS = frozenset().union(*AXIS_REWARD_TERMS, SHARED_REWARD_TERMS)


def axis_permutation() -> tuple[int, int, int]:
    """Which reward group feeds which axis STREAM (scrambled-mapping control, Table IV).

    Default (0, 1, 2) is the correct mapping: the stream that drives the X action is
    credited with X's reward. `AXIS_CREDIT_PERMUTATION=cyclic` rotates it to (1, 2, 0), so
    the X stream is optimised by Y's error while still controlling the X action. Reward
    magnitudes, the number of streams and every other part of the pipeline are unchanged —
    only the assignment is wrong.

    This is the control for "maybe splitting credit into three streams helps on its own,
    regardless of whether the split matches the axes". Without it, the method's claim is
    decomposition; with it, the claim can be attribution.
    """
    from config.policy import CREDIT_DEFAULTS

    mode = CREDIT_DEFAULTS["axis_credit_permutation"]
    if mode in ("", "none", "identity", "0"):
        return (0, 1, 2)
    if mode == "cyclic":
        return (1, 2, 0)
    if mode == "reverse":
        return (2, 1, 0)
    raise RuntimeError(f"Unknown AXIS_CREDIT_PERMUTATION={mode!r} (none|cyclic|reverse)")


def validate_reward_terms(term_names: Sequence[str]) -> None:
    names = frozenset(str(name) for name in term_names)
    unknown = sorted(names - KNOWN_REWARD_TERMS)
    missing = sorted(REQUIRED_REWARD_TERMS - names)
    if unknown or missing:
        raise RuntimeError(
            "Decomposed advantage reward mapping mismatch: "
            f"unknown={unknown or []}, missing={missing or []}"
        )


def decompose_step_reward(term_names: Sequence[str], step_reward: torch.Tensor) -> torch.Tensor:
    """Return [num_envs, 3]; shared terms are copied in full to every axis."""
    validate_reward_terms(term_names)
    if step_reward.ndim != 2 or step_reward.shape[1] != len(term_names):
        raise RuntimeError(
            f"Invalid RewardManager step reward shape {tuple(step_reward.shape)} for {len(term_names)} terms"
        )
    by_name = {str(name): step_reward[:, idx] for idx, name in enumerate(term_names)}
    shared = sum((by_name[name] for name in SHARED_REWARD_TERMS if name in by_name), start=torch.zeros_like(step_reward[:, 0]))
    perm = axis_permutation()
    streams = [shared + sum((by_name[name] for name in AXIS_REWARD_TERMS[perm[axis]] if name in by_name),
                            start=torch.zeros_like(shared))
               for axis in range(len(AXIS_REWARD_TERMS))]
    return torch.stack(streams, dim=1)


def reward_scene_env(vec_env):
    """IsaacLab env owning the RewardManager behind an SB3 env/wrapper chain."""
    env = find_isaac_env(vec_env, required_attrs=("reward_manager", "num_envs"))
    if env is None:
        raise RuntimeError("could not resolve the IsaacLab environment owning reward_manager")
    return env


def manager_step_reward(env) -> tuple[tuple[str, ...], torch.Tensor]:
    """(term names, [N, terms] weighted step reward) from the env's RewardManager."""
    manager = getattr(env, "reward_manager", None)
    terms = tuple(getattr(manager, "_term_names", ()) or ())
    step_reward = getattr(manager, "_step_reward", None)
    if not torch.is_tensor(step_reward):
        raise RuntimeError("RewardManager._step_reward is unavailable")
    return terms, step_reward


def decompose_manager_reward(term_names, step_reward, step_dt, units):
    """Convert Isaac weighted reward rates to PPO per-step units, then split into XYZ streams."""
    validate_reward_units(units)
    if units == "per_step":
        if step_dt is None or not math.isfinite(float(step_dt)) or float(step_dt) <= 0:
            raise ValueError(f"per_step decomposed reward requires positive finite step_dt, got {step_dt!r}")
        step_reward = step_reward * float(step_dt)
    return decompose_step_reward(term_names, step_reward)
