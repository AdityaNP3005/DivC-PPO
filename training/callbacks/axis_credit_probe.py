"""Diagnostic: does the single advantage of joint credit actually track the Z axis?

`fig1b` of the precision report plots the per-axis advantage spread, but that quantity
only exists on decomposed-credit runs: a joint-credit run has one advantage, not three,
so there is nothing per-axis to log. That is structural, not a gap in the logging, and it
leaves the mechanism claim resting on the reward side (`fig1a`) alone.

This probe measures the missing half ON A JOINT RUN. It correlates the run's single
advantage against each axis's share of the step reward:

    corr(A, r_x)   corr(A, r_y)   corr(A, r_z)

If the joint advantage is dominated by Z, corr with r_z is high and corr with r_x / r_y
is low — measured on the joint architecture itself rather than inferred from the method.

Off by default (`AXIS_CREDIT_PROBE=1` to enable) so no existing run changes behaviour.
Implemented as a CALLBACK because joint-credit arms run stock SB3 PPO,
and Asymmetric AC — the matched-pair partner of Our Method — is one of them.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_env.policy.axis_rewards import (
    decompose_step_reward,
    manager_step_reward,
    reward_scene_env,
    validate_reward_terms,
)

_AXES = ("x", "y", "z")


class AxisCreditProbeCallback(BaseCallback):
    """Log corr(advantage, per-axis reward) each rollout. Joint-credit runs only."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._env = None
        self._steps: list[np.ndarray] = []
        self._announced = False
        self._disabled = False

    # ── collection ───────────────────────────────────────────────────────────
    def _on_rollout_start(self) -> None:
        self._steps.clear()

    def _on_step(self) -> bool:
        """One call per buffer row.

        SB3's collect_rollouts fires on_step() after env.step() and BEFORE
        rollout_buffer.add(), so appending here keeps this list in lockstep with the
        buffer's (n_steps, n_envs) layout. Anything that changes that order silently
        turns the correlation into noise, so the alignment is asserted at rollout end.
        """
        if self._disabled:
            return True
        try:
            reward = self._axis_reward()
        except Exception as exc:  # a probe must never take a training run down
            print(f"[AXIS-PROBE] disabled: {exc}")
            self._disabled = True
            return True
        self._steps.append(reward)
        return True

    def _axis_reward(self) -> np.ndarray:
        if self._env is None:
            self._env = reward_scene_env(self.model.env)
        terms, step_reward = manager_step_reward(self._env)
        if not self._announced:
            validate_reward_terms(terms)
            print(f"[AXIS-PROBE] active | axes=XYZ | terms={len(terms)}")
            self._announced = True
        reward = decompose_step_reward(terms, step_reward).detach().cpu().numpy()
        # The buffer stores VecNormalize-scaled rewards and the advantage is built from
        # them, so the per-axis stream has to carry the same scaling or the correlation
        # would mix two units.
        vec = getattr(self.model, "_vec_normalize_env", None)
        if vec is not None and getattr(vec, "norm_reward", False):
            reward = vec.normalize_reward(reward)
        return np.asarray(reward, dtype=np.float32)

    # ── reporting ────────────────────────────────────────────────────────────
    def _on_rollout_end(self) -> None:
        if self._disabled or not self._steps:
            return
        adv = np.asarray(self.model.rollout_buffer.advantages)
        if adv.ndim == 3:
            # decomposed run: each axis already has its OWN advantage, so a
            # "which axis dominates the one advantage" question does not apply
            if not getattr(self, "_skip_announced", False):
                print("[AXIS-PROBE] decomposed credit detected — probe is a JOINT-credit "
                      "measurement, nothing logged")
                self._skip_announced = True
            self._steps.clear()
            return

        rew = np.stack(self._steps, axis=0)           # (n_steps, n_envs, 3)
        if rew.shape[:2] != adv.shape:
            print(f"[AXIS-PROBE] shape mismatch adv{adv.shape} vs reward{rew.shape[:2]} "
                  "— skipping this rollout")
            self._steps.clear()
            return

        a = adv.reshape(-1)
        self.logger.record("train/advantage_std", float(a.std()))
        for i, axis in enumerate(_AXES):
            r = rew[:, :, i].reshape(-1)
            self.logger.record(f"train/adv_corr_{axis}", _corr(a, r))
            self.logger.record(f"train/axis_reward_std_{axis}", float(r.std()))
        self._steps.clear()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, 0.0 when either side is constant (np.corrcoef would return nan)."""
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
