from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class AxisRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class AxisRolloutBuffer(RolloutBuffer):
    """SB3 rollout buffer with independent XYZ reward/value/log-prob streams."""

    axis_dim = 3

    def _axes(self, value) -> np.ndarray:
        """(n_envs, 3) view of a critic output. A shared (scalar) critic emits one column,
        which becomes the SAME baseline on every axis — see config.policy scalar_critic."""
        arr = np.asarray(value).reshape(self.n_envs, -1)
        return np.broadcast_to(arr, (self.n_envs, self.axis_dim)) if arr.shape[1] == 1 else arr

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs, self.axis_dim)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values_np = self._axes(last_values.detach().cpu().numpy())
        last_gae = np.zeros((self.n_envs, self.axis_dim), dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            mask = next_non_terminal[:, None]
            delta = self.rewards[step] + self.gamma * next_values * mask - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[self.pos] = np.asarray(obs)
        self.actions[self.pos] = np.asarray(action)
        self.rewards[self.pos] = np.asarray(reward).reshape(self.n_envs, self.axis_dim)
        self.episode_starts[self.pos] = np.asarray(episode_start)
        self.values[self.pos] = self._axes(value.detach().cpu().numpy())
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy().reshape(self.n_envs, self.axis_dim)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[AxisRolloutBufferSamples, None, None]:
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns"):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        batch_size = batch_size or self.buffer_size * self.n_envs
        for start in range(0, self.buffer_size * self.n_envs, batch_size):
            yield self._get_samples(indices[start : start + batch_size])

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> AxisRolloutBufferSamples:
        return AxisRolloutBufferSamples(
            observations=self.to_torch(self.observations[batch_inds]),
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds]),
            old_log_prob=self.to_torch(self.log_probs[batch_inds]),
            advantages=self.to_torch(self.advantages[batch_inds]),
            returns=self.to_torch(self.returns[batch_inds]),
        )
