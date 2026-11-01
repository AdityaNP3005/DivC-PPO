"""Scalar PPO learner units with the same per-step return RMS as DivC.

The environment still returns per-step scalar rewards. Only the reward presented
to the scalar learner is converted to a rate, before normalization and clipping.
DivC continues using its existing collector and ordinary VecNormalize unchanged.
"""
import math

from stable_baselines3.common.vec_env import VecNormalize

from config.reward_units import validate_reward_units


class PpoRewardVecNormalize(VecNormalize):
    def __init__(self, venv, *, reward_units, step_dt, **kwargs):
        self.scalar_reward_units = validate_reward_units(reward_units)
        self.reward_step_dt = float(step_dt)
        if not math.isfinite(self.reward_step_dt) or self.reward_step_dt <= 0:
            raise ValueError("Scalar PPO reward conversion requires positive finite step_dt")
        super().__init__(venv, **kwargs)

    def normalize_reward(self, reward):
        # step_wait updates ret_rms from the unmodified environment reward first.
        # Inference disables norm_reward and receives the original env reward.
        if self.norm_reward and self.scalar_reward_units == "legacy_rate":
            reward = reward / self.reward_step_dt
        return super().normalize_reward(reward)
