from __future__ import annotations

"""Single-actor PPO with independent XYZ credit (DivC-PPO)."""

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from rl_env.policy.axis_rewards import (
    decompose_manager_reward,
    manager_step_reward,
    reward_scene_env,
    validate_reward_terms,
)
from config.reward_units import reward_units_of
from rl_env.policy.axis_rollout_buffer import AxisRolloutBuffer


class DivCPPO(PPO):
    """PPO with independent XYZ advantages for single-actor policies."""

    _axis_env: Any = None
    _axis_mapping_announced = False

    def __init__(self, *args, **kwargs):
        policy_kwargs = kwargs.get("policy_kwargs") or {}
        axis_cfg = policy_kwargs.get("decomposed_config") or {}
        if bool(axis_cfg.get("decomposed_advantage", False)):
            kwargs.setdefault("rollout_buffer_class", AxisRolloutBuffer)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_axis_env", "_axis_mapping_announced"]

    def _credit_cfg(self) -> dict[str, Any]:
        cfg = getattr(self.policy, "decomposed_config", None)
        return cfg if isinstance(cfg, dict) else {}

    def _decomposed(self) -> bool:
        return bool(self._credit_cfg().get("decomposed_advantage", False))


    def _axis_rewards(self) -> np.ndarray:
        if self._axis_env is None:
            self._axis_env = reward_scene_env(self.env)
        terms, step_reward = manager_step_reward(self._axis_env)
        if not self._axis_mapping_announced:
            validate_reward_terms(terms)
            print(f"[DECOMPOSED_ADV] reward mapping active | axes=XYZ | terms={len(terms)}")
            self._axis_mapping_announced = True
        reward = decompose_manager_reward(
            terms, step_reward, getattr(self._axis_env, "step_dt", None), reward_units_of(self._credit_cfg())
        ).detach().cpu().numpy()
        if self._vec_normalize_env is not None and self._vec_normalize_env.norm_reward:
            reward = self._vec_normalize_env.normalize_reward(reward)
        return reward.astype(np.float32, copy=False)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        if not self._decomposed():
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not isinstance(rollout_buffer, AxisRolloutBuffer):
            raise RuntimeError("Decomposed advantage requires AxisRolloutBuffer")
        if not isinstance(self.action_space, spaces.Box) or int(np.prod(self.action_space.shape)) != 3:
            raise RuntimeError("Decomposed advantage requires a 3D Cartesian Box action space")
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        n_steps = 0
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_prob_axes = self.policy.forward_decomposed(obs_tensor)
            actions_np = actions.cpu().numpy()
            clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            axis_rewards = self._axis_rewards()
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_values = self.policy.predict_values(terminal_obs)[0]
                    axis_rewards[idx] += self.gamma * terminal_values.detach().cpu().numpy()
            rollout_buffer.add(
                self._last_obs,
                actions_np,
                axis_rewards,
                self._last_episode_starts,
                values,
                log_prob_axes,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        if self._decomposed():
            self._train_decomposed()
        else:
            super().train()

    def _train_decomposed(self) -> None:
        if not isinstance(self.rollout_buffer, AxisRolloutBuffer):
            raise RuntimeError("Decomposed advantage requires AxisRolloutBuffer")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None if self.clip_range_vf is None else self.clip_range_vf(self._current_progress_remaining)
        weights = th.as_tensor(
            self._credit_cfg().get("decomposed_axis_weights", (1.0, 1.0, 1.0)),
            device=self.device,
            dtype=th.float32,
        )
        # Shared-ratio control (Table IV): read once per train() call, not per minibatch.
        shared_ratio = bool(self._credit_cfg().get("shared_ratio", False))
        pg_axes_log: list[np.ndarray] = []
        vf_axes_log: list[np.ndarray] = []
        kl_axes_log: list[np.ndarray] = []
        clip_axes_log: list[np.ndarray] = []
        entropy_log: list[float] = []
        continue_training = True
        loss = th.zeros((), device=self.device)
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values, log_prob_axes, entropy_axes = self.policy.evaluate_actions_decomposed(
                    data.observations, data.actions
                )
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)
                ratio = th.exp(log_prob_axes - data.old_log_prob)
                if shared_ratio:
                    # ONE ratio for the whole action: the joint log-prob ratio is the SUM
                    # over axes, so a large step on any axis moves it and the clip then
                    # truncates all three streams together. Everything upstream (per-axis
                    # reward, GAE, normalisation) is untouched.
                    ratio = th.exp(
                        (log_prob_axes - data.old_log_prob).sum(dim=1, keepdim=True)
                    ).expand_as(ratio)
                loss_1 = advantages * ratio
                loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_axes = -th.min(loss_1, loss_2).mean(dim=0)
                policy_loss = th.sum(weights * pg_axes)
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = data.old_values + th.clamp(
                        values - data.old_values, -clip_range_vf, clip_range_vf
                    )
                vf_axes = th.mean((data.returns - values_pred) ** 2, dim=0)
                value_loss = vf_axes.mean()
                entropy_loss = -entropy_axes.sum(dim=1).mean()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                with th.no_grad():
                    log_ratio = log_prob_axes - data.old_log_prob
                    kl_axes = th.mean((th.exp(log_ratio) - 1) - log_ratio, dim=0)
                    joint_kl = float(kl_axes.sum().item())
                    clip_axes = th.mean((th.abs(ratio - 1) > clip_range).float(), dim=0)
                if self.target_kl is not None and joint_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                pg_axes_log.append(pg_axes.detach().cpu().numpy())
                vf_axes_log.append(vf_axes.detach().cpu().numpy())
                kl_axes_log.append(kl_axes.cpu().numpy())
                clip_axes_log.append(clip_axes.cpu().numpy())
                entropy_log.append(float(entropy_loss.detach().item()))
            self._n_updates += 1
            if not continue_training:
                break
        self._log_decomposed_train(
            loss, clip_range, pg_axes_log, vf_axes_log, kl_axes_log, clip_axes_log,
            entropy_log,
        )

    def _log_decomposed_train(
        self, loss, clip_range, pg, vf, kl, clip, entropy
    ) -> None:
        labels = ("x", "y", "z")
        pg_mean = np.mean(pg, axis=0)
        vf_mean = np.mean(vf, axis=0)
        kl_mean = np.mean(kl, axis=0)
        clip_mean = np.mean(clip, axis=0)
        rewards = np.mean(self.rollout_buffer.rewards, axis=(0, 1))
        advantages = np.asarray(self.rollout_buffer.advantages).reshape(-1, 3)
        for i, axis in enumerate(labels):
            self.logger.record(f"train_axis/{axis}_policy_loss", float(pg_mean[i]))
            self.logger.record(f"train_axis/{axis}_value_loss", float(vf_mean[i]))
            self.logger.record(f"train_axis/{axis}_approx_kl", float(kl_mean[i]))
            self.logger.record(f"train_axis/{axis}_clip_fraction", float(clip_mean[i]))
            self.logger.record(f"train_axis/{axis}_reward_mean", float(rewards[i]))
            self.logger.record(f"train_axis/{axis}_advantage_std", float(advantages[:, i].std()))
        self.logger.record("train/policy_gradient_loss", float(np.sum(pg_mean)))
        self.logger.record("train/value_loss", float(np.mean(vf_mean)))
        self.logger.record("train/approx_kl", float(np.sum(kl_mean)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_mean)))
        self.logger.record("train/entropy_loss", float(np.mean(entropy)))
        self.logger.record("train/loss", float(loss.detach().item()))
        ev = [explained_variance(self.rollout_buffer.values[:, i], self.rollout_buffer.returns[:, i]) for i in range(3)]
        self.logger.record("train/explained_variance", float(np.nanmean(ev)))
        self._log_common(clip_range)


    def _log_common(self, clip_range: float) -> None:
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self.clip_range_vf(self._current_progress_remaining))
