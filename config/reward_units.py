"""Reward units/source contracts, safe before Isaac launches.

Fresh runs use the legacy-rate learner units of the paper checkpoints.
Missing checkpoint credit metadata keeps the historical legacy-rate collector.
"""
import os


def validate_reward_units(value):
    if value not in ("per_step", "legacy_rate"):
        raise ValueError(f"Invalid reward_units={value!r}: per_step|legacy_rate")
    return value


def fresh_reward_units():
    return validate_reward_units((os.environ.get("DIVC_REWARD_UNITS") or "legacy_rate").strip())


def reward_units_of(config):
    return validate_reward_units((config or {}).get("reward_units", "legacy_rate"))


def scalar_reward_units_of(config):
    return validate_reward_units((config or {}).get("scalar_reward_units", "per_step"))


def reward_error_source(asymmetric_critic):
    source = (os.environ.get("REWARD_ERROR_SOURCE") or "auto").strip().lower()
    if source == "auto":
        return "gt"  # the paper trains both methods on the simulator-state reward
    if source not in ("gt", "vision"):
        raise ValueError("REWARD_ERROR_SOURCE must be auto|gt|vision")
    return source
