from __future__ import annotations

"""Single-actor PPO/DivC-PPO credit and critic settings."""

from typing import Any, Optional
import os
from .env_flags import env_bool, env_scale3
from .control import validate_cartesian_metadata
from .reward_units import fresh_reward_units

def scalar_critic_of(cfg: Optional[dict[str, Any]]) -> bool:
    """Is the value head a single scalar broadcast to all three axes?

    THE one place that knows `scalar_critic` used to be called `shared_critic`. Checkpoints,
    `agent.yaml` and old shell scripts still carry the old key, and anything older than the
    flag carries neither — which meant the 3-dim head, hence the False fallback (NOT the
    config default below, which is True: rebuilding a pre-flag checkpoint 1-dim would fail
    to load its weights).

    Lives HERE, not in rl_env.policy, because training/runtime_args.py needs it and that
    module must stay importable before AppLauncher — importing rl_env.policy would pull in
    torch + stable_baselines3 ahead of Isaac Sim's launcher and break the run.
    """
    src = cfg or {}
    if "scalar_critic" in src:
        return bool(src["scalar_critic"])
    return bool(src.get("shared_critic", False))

ASYMMETRIC_CRITIC_DEFAULTS = {"privileged_obs_dim": 6}

CREDIT_DEFAULTS = {
    "reward_units": fresh_reward_units(),
    "decomposed_advantage": env_bool("DECOMPOSED_ADVANTAGE", True),
    "decomposed_axis_weights": env_scale3("DECOMPOSED_AXIS_WEIGHTS", (1.0, 1.0, 1.0)),
    "scalar_critic": env_bool("SCALAR_CRITIC", env_bool("SHARED_CRITIC", True)),
    "shared_ratio": env_bool("SHARED_RATIO", False),
    "axis_credit_permutation": (os.environ.get("AXIS_CREDIT_PERMUTATION") or "none").strip().lower(),
}

def validate_supported_architecture(cfg: dict) -> None:
    """Reject removed architectures before importing/loading their policy classes."""
    validate_cartesian_metadata(cfg)
    kwargs = cfg.get("policy_kwargs", {}) or {}
    if str(cfg.get("ppo_head", cfg.get("PPO_head", "single"))).lower() != "single" or kwargs.get("dual_head_config"):
        raise ValueError("Only single-actor checkpoints are supported; dual-action checkpoints are unsupported.")
    if str(cfg.get("z_control_mode", "policy")).lower() != "policy":
        raise ValueError("Policy-controlled XYZ is required; scripted-Z checkpoints are unsupported.")
    if str(cfg.get("gate_mode", "rule")).lower() == "learned":
        raise ValueError("Learned-gate checkpoints are unsupported.")

def checkpoint_actor_depth(cfg: dict) -> bool:
    """Keep the observation width of supported single-actor legacy checkpoints."""
    return bool(cfg.get("actor_depth_enabled", cfg.get("gate_z_enabled", False)))


def publish_run_mode(args) -> None:
    """Export the flags rl_env.env.runtime reads when the env config is imported (training and validation)."""
    os.environ["RL_CONTROL_MODE"] = args.mode
    os.environ["ASYMMETRIC_CRITIC"] = "1" if args.asymmetric_critic else "0"
    os.environ["ACTOR_DEPTH_ENABLED"] = "1" if args.actor_depth_enabled else "0"
