"""Per-term PPO reward logging to TensorBoard (episode-end aggregation)."""

import torch
from rl_env.env_wrappers import training_scene_env
from stable_baselines3.common.callbacks import BaseCallback


_LEGACY_AXIS_GROUPS = {
    "gt_idle": ("gt_idle_x", "gt_idle_y", "gt_idle_z"),
    "action_rate": ("action_rate_x", "action_rate_y", "action_rate_z"),
}


def _with_legacy_axis_sums(values: dict):
    out = dict(values)
    for legacy, axes in _LEGACY_AXIS_GROUPS.items():
        if all(axis in values for axis in axes):
            out[legacy] = values[axes[0]] + values[axes[1]] + values[axes[2]]
    return out


class RewardTermLoggerCallback(BaseCallback):
    """Log per-term PPO reward values to TensorBoard."""

    def __init__(self, log_interval_steps: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self._isaaclab_env = None
        self._log_interval_steps = int(log_interval_steps)
        self._last_logged_steps = -1
        self._warned_reward_logger = False
        self._source_printed = False
        # Per-episode reward aggregation (captured at each episode END, not a
        # single-step snapshot). Impulse terms like success_once are invisible in a
        # snapshot; here we accumulate each completed episode's term-sums.
        self._ep_rew_sum: dict[str, float] = {}   # Σ over completed episodes of each term's episode-sum
        self._ep_rew_count: int = 0               # number of episodes completed since last log
        self._ep_len_sum: float = 0.0             # Σ lengths of those episodes (for per-step mean)
        self._ep_success_count: int = 0           # episodes whose success_once fired (>0)
        self._prev_ep_sums: dict | None = None    # previous step's _episode_sums (pre-reset)
        self._prev_ep_len = None                  # previous step's episode_length_buf

    def _get_isaaclab_env(self):
        return training_scene_env(self.training_env)

    def _on_training_start(self) -> None:
        try:
            self._isaaclab_env = self._get_isaaclab_env()
        except Exception as e:
            print(f"[RewardLogger] Could not resolve the IsaacLab env: {e}")

    def _accumulate_episode_rewards(self) -> None:
        """Capture each completed episode's per-term reward SUM at episode end.

        IsaacLab zeroes reward_manager._episode_sums for envs it resets, so (like
        _accumulate_gt_stats) we keep the previous step's sums and harvest them for
        envs whose episode_length_buf just dropped (episode boundary).
        """
        if self._isaaclab_env is None:
            try:
                self._isaaclab_env = self._get_isaaclab_env()
            except Exception:
                return
        rm = getattr(self._isaaclab_env, "reward_manager", None)
        ep_len = getattr(self._isaaclab_env, "episode_length_buf", None)
        if rm is None or ep_len is None:
            return
        ep_sums = getattr(rm, "_episode_sums", None)
        if not isinstance(ep_sums, dict) or not ep_sums:
            return
        ep_len = ep_len.detach()
        cur = {k: v.detach().clone() for k, v in ep_sums.items()}

        prev, prev_len = self._prev_ep_sums, self._prev_ep_len
        if prev is not None and prev_len is not None and prev_len.shape == ep_len.shape:
            just_reset = ep_len < prev_len
            n = int(just_reset.sum().item())
            if n > 0:
                for term, vals in prev.items():
                    if vals.shape[0] != just_reset.shape[0]:
                        continue
                    self._ep_rew_sum[term] = self._ep_rew_sum.get(term, 0.0) + float(vals[just_reset].sum().item())
                self._ep_rew_count += n
                self._ep_len_sum += float(prev_len[just_reset].float().sum().item())
                succ = prev.get("success_once", None)
                if succ is not None and succ.shape[0] == just_reset.shape[0]:
                    self._ep_success_count += int((succ[just_reset] > 0).sum().item())

        self._prev_ep_sums = cur
        self._prev_ep_len = ep_len.clone()

    def _log_episode_rewards(self) -> None:
        """Report per-episode reward stats accumulated since the last log, then reset."""
        count = self._ep_rew_count
        avg_len = max(self._ep_len_sum / count, 1.0)
        for term, total in _with_legacy_axis_sums(self._ep_rew_sum).items():
            mean_sum = total / count
            # Per-step mean over the episode (familiar scale) + the per-episode total
            # (impulse terms like success_once are visible here: ep_sum ≈ weight × rate).
            self.logger.record(f"rewards/{term}", mean_sum / avg_len)
            self.logger.record(f"rewards/{term}_ep_sum", mean_sum)
        # This is reward-source reach (vision for v1/v2, GT for v3-v5), not the
        # fair stay-20 success metric logged by TrainingMonitorCallback.
        self.logger.record("task/reward_reach_rate", self._ep_success_count / count)
        self.logger.record("rewards/episodes_counted", float(count))
        if not self._source_printed:
            print("[RewardLogger] Per-episode reward logging on (rewards/* = mean per-step over completed "
                  "episodes; rewards/*_ep_sum = per-episode total; task/reward_reach_rate added).")
            self._source_printed = True
        self._ep_rew_sum = {}
        self._ep_rew_count = 0
        self._ep_len_sum = 0.0
        self._ep_success_count = 0

    def _log_reward_terms(self) -> None:
        if self._isaaclab_env is None:
            try:
                self._isaaclab_env = self._get_isaaclab_env()
            except Exception:
                return

        current_steps = int(getattr(self.model, "num_timesteps", 0))
        if self._last_logged_steps >= 0 and current_steps - self._last_logged_steps < self._log_interval_steps:
            return
        self._last_logged_steps = current_steps

        # Prefer per-episode aggregation (captured at episode end). The single-step
        # snapshot below is only a fallback before any episode has completed.
        if self._ep_rew_count > 0:
            self._log_episode_rewards()
            return

        try:
            rm = getattr(self._isaaclab_env, "reward_manager", None)
            if rm is None:
                if not self._warned_reward_logger:
                    print("[RewardLogger] reward_manager not found; per-term reward logging skipped.")
                    self._warned_reward_logger = True
                return

            logged_any = False
            source_name = None

            def _to_tensor(value):
                if torch.is_tensor(value):
                    return value.detach().float()
                try:
                    return torch.as_tensor(value, device=getattr(self._isaaclab_env, "device", None)).detach().float()
                except Exception:
                    return None

            def _safe_name(name: object) -> str:
                return str(name).replace("/", "_").replace(" ", "_")

            def _record_step(name: object, values) -> bool:
                vals = _to_tensor(values)
                if vals is None or vals.numel() == 0:
                    return False
                flat = vals.reshape(-1)
                key = _safe_name(name)
                self.logger.record(f"rewards/{key}", flat.mean().item())
                self.logger.record(f"rewards/{key}_abs_max", flat.abs().max().item())
                return True

            def _record_episode(name: object, values, denom=None) -> bool:
                vals = _to_tensor(values)
                if vals is None or vals.numel() == 0:
                    return False
                flat = vals.reshape(-1)
                key = _safe_name(name)
                self.logger.record(f"rewards_episode/{key}_sum_mean", flat.mean().item())
                self.logger.record(f"rewards_episode/{key}_sum_abs_max", flat.abs().max().item())
                if denom is not None and vals.shape[:1] == denom.shape[:1]:
                    per_env = vals.reshape(denom.shape[0], -1).mean(dim=1) / denom
                    self.logger.record(f"rewards/{key}_per_step", per_env.mean().item())
                return True

            step_reward = getattr(rm, "_step_reward", None)
            term_names = getattr(rm, "_term_names", None)
            step_t = _to_tensor(step_reward)
            if step_t is not None and step_t.ndim >= 2 and term_names is not None:
                step_by_name = {}
                for idx, term_name in enumerate(term_names):
                    if idx < step_t.shape[1]:
                        logged_any |= _record_step(term_name, step_t[:, idx])
                        step_by_name[str(term_name)] = step_t[:, idx]
                for legacy, values in _with_legacy_axis_sums(step_by_name).items():
                    if legacy not in step_by_name:
                        logged_any |= _record_step(legacy, values)
                logged_any |= _record_step("total_per_step", step_t.sum(dim=-1))
                source_name = "reward_manager._step_reward"

            if not logged_any:
                episode_sums = getattr(rm, "_episode_sums", None)
                episode_len = getattr(self._isaaclab_env, "episode_length_buf", None)
                denom = None
                if episode_len is not None and torch.is_tensor(episode_len):
                    denom = torch.clamp(episode_len.detach().float(), min=1.0)

                if isinstance(episode_sums, dict) and episode_sums:
                    total = None
                    for term_name, term_values in _with_legacy_axis_sums(episode_sums).items():
                        logged_any |= _record_episode(term_name, term_values, denom)
                        vals = _to_tensor(term_values)
                        if vals is not None and term_name not in _LEGACY_AXIS_GROUPS:
                            total = vals if total is None else total + vals
                    if total is not None:
                        logged_any |= _record_episode("total", total, denom)
                    source_name = "reward_manager._episode_sums"
                else:
                    sums_t = _to_tensor(episode_sums)
                    if sums_t is not None and sums_t.ndim >= 2 and term_names is not None:
                        for idx, term_name in enumerate(term_names):
                            if idx < sums_t.shape[1]:
                                logged_any |= _record_episode(term_name, sums_t[:, idx], denom)
                        logged_any |= _record_episode("total", sums_t.sum(dim=-1), denom)
                        source_name = "reward_manager._episode_sums"

            if not logged_any:
                extras = getattr(self._isaaclab_env, "extras", {}) or {}
                log_data = extras.get("log", {}) if isinstance(extras, dict) else {}
                if isinstance(log_data, dict):
                    for key, value in log_data.items():
                        if "reward" in str(key).lower():
                            logged_any |= _record_step(key, value)
                    source_name = "env.extras['log']"

            if logged_any and not self._source_printed:
                print(f"[RewardLogger] Per-term reward logging on: source={source_name}")
                self._source_printed = True
            elif not logged_any and not self._warned_reward_logger:
                print("[RewardLogger] No reward term source found; TensorBoard per-term reward logging skipped.")
                self._warned_reward_logger = True
        except Exception as e:
            if not self._warned_reward_logger:
                print(f"[RewardLogger] Failed to log reward terms: {e}")
                self._warned_reward_logger = True

    def _on_rollout_end(self) -> None:
        self._log_reward_terms()

    def _on_step(self) -> bool:
        self._accumulate_episode_rewards()
        return True
