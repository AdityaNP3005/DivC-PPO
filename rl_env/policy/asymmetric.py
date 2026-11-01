from __future__ import annotations

"""AsymmetricActorCriticPolicy: actor sees obs[0:actor_obs_dim], critic sees the
ee_delta prefix + privileged GT tail (zero-masked slices of one shared observation)."""

from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from config.policy import scalar_critic_of


def _mask_privileged_features(features: th.Tensor, actor_obs_dim: Optional[int]) -> th.Tensor:
    """Zero out the privileged TAIL so actor only sees deployable features."""
    if actor_obs_dim is None or features.shape[-1] <= int(actor_obs_dim):
        return features
    masked = features.clone()
    masked[..., int(actor_obs_dim):] = 0.0
    return masked


def _mask_actor_features(features: th.Tensor, actor_obs_dim: Optional[int], keep_prefix: int = 3) -> th.Tensor:
    """Zero out visual actor features [keep_prefix:actor_obs_dim] so critic
    only sees ee_delta (first keep_prefix) + privileged tail.
    """
    if actor_obs_dim is None or features.shape[-1] <= int(actor_obs_dim):
        return features
    masked = features.clone()
    start = min(int(keep_prefix), int(actor_obs_dim))
    masked[..., start:int(actor_obs_dim)] = 0.0
    return masked


def _normalize_actor_obs_dim(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        dim = int(value)
    except Exception:
        return None
    return dim if dim > 0 else None


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy where actor sees only the first actor_obs_dim features.

    The full observation still goes to the critic. This is a lightweight
    privileged-critic setup compatible with SB3 VecNormalize because the actor
    receives a constant zero tail in normalized feature space.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_obs_dim: Optional[int] = None,
        decomposed_config: Optional[dict[str, Any]] = None,
    ):
        self.actor_obs_dim = _normalize_actor_obs_dim(actor_obs_dim)
        self.decomposed_config = dict(decomposed_config or {})
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                actor_obs_dim=self.actor_obs_dim,
                decomposed_config=dict(self.decomposed_config),
            )
        )
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # scalar_critic (the default) leaves SB3's scalar value head in place under
        # per-axis credit; the buffer broadcasts it to all three axes. decomposed_config is
        # the RAW dict from the checkpoint here (no normalize step), so the legacy key has
        # to be honoured through scalar_critic_of.
        if not bool(
            self.decomposed_config.get("decomposed_advantage", False)
        ) or scalar_critic_of(self.decomposed_config):
            return
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 3)
        if self.ortho_init:
            self.value_net.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def set_actor_obs_dim(self, actor_obs_dim: Optional[int]) -> None:
        self.actor_obs_dim = _normalize_actor_obs_dim(actor_obs_dim)

    def _actor_features(self, features: th.Tensor) -> th.Tensor:
        return _mask_privileged_features(features, self.actor_obs_dim)

    def _critic_features(self, features: th.Tensor) -> th.Tensor:
        return _mask_actor_features(features, self.actor_obs_dim)

    def _actor_critic_latents(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi = self.mlp_extractor.forward_actor(self._actor_features(features))
            latent_vf = self.mlp_extractor.forward_critic(self._critic_features(features))
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(self._actor_features(pi_features))
            latent_vf = self.mlp_extractor.forward_critic(self._critic_features(vf_features))
        return latent_pi, latent_vf

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._actor_critic_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def forward_decomposed(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._actor_critic_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob_axes = distribution.distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob_axes

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        latent_pi, latent_vf = self._actor_critic_latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def evaluate_actions_decomposed(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._actor_critic_latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return (
            self.value_net(latent_vf),
            distribution.distribution.log_prob(actions),
            distribution.distribution.entropy(),
        )

    def get_distribution(self, obs: PyTorchObs):
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(self._actor_features(features))
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(self._critic_features(features))
        return self.value_net(latent_vf)

    def asymmetric_debug_stats(self) -> dict[str, float]:
        obs_dim = int(np.prod(getattr(self.observation_space, "shape", (0,))))
        actor_dim = obs_dim if self.actor_obs_dim is None else min(int(self.actor_obs_dim), obs_dim)
        priv_dim = max(obs_dim - actor_dim, 0)
        return {
            "actor_obs_dim": float(actor_dim),
            "critic_obs_dim": float(obs_dim if priv_dim == 0 else 3 + priv_dim),
            "privileged_obs_dim": float(priv_dim),
        }
