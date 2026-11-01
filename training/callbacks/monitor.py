"""Terminal training monitor callback (lifecycle + metric collection + printing).

Per-rollout console summary of GT centering, YOLO detection, crane error
and ETA. The GT/YOLO/crane accumulation lives in _MonitorStatsMixin
(monitor_stats.py); this file keeps __init__, the SB3 lifecycle hooks, metric
collection and the printed table.
"""

import time

from rl_env.env_wrappers import training_scene_env

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .monitor_stats import _MonitorStatsMixin
from .monitor_print import _MonitorPrintMixin


# ─── Monitoring Callback ─────────────────────────────────────────────────────
class TrainingMonitorCallback(_MonitorStatsMixin, _MonitorPrintMixin, BaseCallback):
    """Print a PPO training summary to the terminal."""

    def __init__(self, model_type: str = "PPO", print_interval_steps: int = 50_000, total_steps: int | None = None, verbose: int = 1):
        super().__init__(verbose)
        self._model_type = model_type
        self._print_interval_steps = print_interval_steps
        self._total_steps = int(total_steps) if total_steps is not None else None
        self._isaaclab_env = None
        self._t_start: float = 0.0
        self._t_last:  float = 0.0
        self._steps_last: int = 0
        self._start_steps: int = 0
        self._eta_t_last: float = 0.0
        self._eta_steps_last: int = 0
        self._eta_rate_ema: float | None = None
        self._rollout_count: int = 0
        self._reset_rollout_totals()
        self._reset_episode_state()

    def _reset_rollout_totals(self) -> None:
        """Clear rollout aggregates without touching in-flight episode state.

        GT/YOLO/crane totals summarize completed episode endpoints. Error sums
        are in meters; hold sums count in-band steps from successful episodes.
        Per-env sticky reach/stay/near buffers belong to _reset_episode_state.
        """
        self._yolo_center_x_sum = 0.0
        self._yolo_center_y_sum = 0.0
        self._yolo_depth_sum = 0.0
        self._yolo_metric_count = 0
        self._yolo_total_env_steps = 0
        self._gt_target_x_sum = 0.0
        self._gt_target_y_sum = 0.0
        self._gt_target_z_sum = 0.0
        self._ee_pos_x_sum = 0.0
        self._ee_pos_y_sum = 0.0
        self._ee_pos_z_sum = 0.0
        self._gt_error_x_sum = 0.0
        self._gt_error_y_sum = 0.0
        self._gt_error_z_sum = 0.0
        self._gt_error_norm_sum = 0.0
        self._gt_metric_count = 0
        self._gt_success_count = 0
        self._gt_reach_count = 0
        self._gt_error_xy_sum = 0.0
        self._gt_error_z_abs_sum = 0.0
        self._hold_steps_sum = 0.0
        self._hold_success_episodes = 0
        self._near_fail_count = 0
        self._crane_err_sum = 0.0
        self._crane_err_count = 0
        self._last_yolo_cache_step = None

    def _reset_episode_state(self) -> None:
        """Initialize per-env sticky success/near and previous-frame buffers.

        Never call at a rollout boundary: episodes can span many rollouts.
        Actual episode resets are handled per-env by the stats collector.
        """
        self._gt_ever_at_target = None
        self._gt_stay_streak = None
        self._gt_stable_success = None
        self._hold_count = None
        self._gt_ever_near = None
        self._ee_body_id = None
        self._last_yolo_has_det = None
        self._last_yolo_centers = None
        self._last_yolo_depths = None
        self._last_crane_err = None
        self._last_episode_length_buf = None

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        self._t_last = self._t_start
        # On resume, SB3 keeps model.num_timesteps from the checkpoint. ETA must
        # count only steps collected in this run, not historical checkpoint steps.
        current_steps = int(getattr(self.model, "num_timesteps", 0))
        self._start_steps = int(getattr(self, "_start_steps", current_steps))
        self._steps_last = current_steps
        self._eta_t_last = self._t_start
        self._eta_steps_last = current_steps
        self._eta_rate_ema = None
        self._rollout_count = 0
        self._reset_rollout_totals()
        self._reset_episode_state()
        try:
            self._isaaclab_env = self._get_isaaclab_env()
        except Exception as e:
            print(f"[Monitor] Could not resolve the IsaacLab env: {e}")
        interval_label = "per rollout"
        print("\n" + "═" * 70)
        print(f"  TRAINING MONITOR  |  num_envs={self.training_env.num_envs}  |  model={self._model_type}  |  log {interval_label}")
        print("═" * 70)

    def _get_isaaclab_env(self):
        return training_scene_env(self.training_env)

    def _collect_rollout_metrics(self) -> dict[str, object]:
        buf = self.model.ep_info_buffer
        metrics = {}
        if len(buf) > 0:
            rews = [ep["r"] for ep in buf]
            lens = [ep["l"] for ep in buf]
            metrics["ep_len_mean"] = float(np.mean(lens))
            metrics["ep_rew_mean"] = float(np.mean(rews))
        if self._yolo_total_env_steps > 0:
            metrics["det_rate"] = float(self._yolo_metric_count) / float(self._yolo_total_env_steps)
        if self._yolo_metric_count > 0:
            ycount = float(self._yolo_metric_count)
            metrics["yolo_center_x_mean_px"] = self._yolo_center_x_sum / ycount
            metrics["yolo_center_y_mean_px"] = self._yolo_center_y_sum / ycount
            metrics["yolo_depth_mean_m"] = self._yolo_depth_sum / ycount
        if self._gt_metric_count > 0:
            count = float(self._gt_metric_count)
            metrics["gt_target_x_mean_m"] = self._gt_target_x_sum / count
            metrics["gt_target_y_mean_m"] = self._gt_target_y_sum / count
            metrics["gt_target_z_mean_m"] = self._gt_target_z_sum / count
            metrics["ee_pos_x_mean_m"] = self._ee_pos_x_sum / count
            metrics["ee_pos_y_mean_m"] = self._ee_pos_y_sum / count
            metrics["ee_pos_z_mean_m"] = self._ee_pos_z_sum / count
            metrics["gt_error_x_mean_m"] = self._gt_error_x_sum / count
            metrics["gt_error_y_mean_m"] = self._gt_error_y_sum / count
            metrics["gt_error_z_mean_m"] = self._gt_error_z_sum / count
            metrics["gt_error_norm_mean_m"] = self._gt_error_norm_sum / count
            # Paper eval metrics: Final Center Error Ēxy, Final Depth Error Ēz, NFR, Precision Hold H̄.
            metrics["gt_error_xy_mean_m"] = self._gt_error_xy_sum / count
            metrics["gt_error_z_abs_mean_m"] = self._gt_error_z_abs_sum / count
            metrics["near_fail_rate"] = float(self._near_fail_count) / count
            if self._hold_success_episodes > 0:
                metrics["precision_hold_steps"] = self._hold_steps_sum / float(self._hold_success_episodes)
            # Fair cross-ablation stable success: 20 consecutive TRUE-GT in-band steps.
            metrics["gt_success_rate"] = float(self._gt_success_count) / count
            metrics["success_rate"] = metrics["gt_success_rate"]
            metrics["gt_reach_rate"] = float(self._gt_reach_count) / count
        if self._crane_err_count > 0:
            metrics["crane_err_mean"] = self._crane_err_sum / float(self._crane_err_count)
        return metrics

    def _collect_train_metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = {}
        try:
            name_to_value = getattr(self.logger, "name_to_value", {}) or {}
            for key, value in name_to_value.items():
                if not str(key).startswith("train/"):
                    continue
                short_key = str(key).split("/", 1)[1]
                try:
                    metrics[short_key] = float(value)
                except Exception:
                    metrics[short_key] = value
        except Exception:
            pass
        return metrics

    def _record_rollout_metrics_to_logger(self, rollout_metrics: dict[str, object]) -> None:
        """Persist custom task metrics to the SB3 logger so they land in TensorBoard.

        Without this, det_rate / crane_err_mean / gt_error_norm only exist in the
        printed table and cannot be replotted afterwards. Logging them under the
        `task/` namespace keeps the plots and training in sync.
        """
        # Map internal metric keys -> stable TensorBoard tags consumed by the plots.
        _TAG_MAP = {
            "det_rate": "task/det_rate",
            "crane_err_mean": "task/crane_err_mean",
            "gt_error_norm_mean_m": "task/gt_error_norm_m",
            "gt_success_rate": "task/gt_success_rate",
            "success_rate": "task/success_rate",
            "gt_reach_rate": "task/gt_reach_rate",
            "gt_error_x_mean_m": "task/gt_error_x_m",
            "gt_error_y_mean_m": "task/gt_error_y_m",
            "gt_error_z_mean_m": "task/gt_error_z_m",
            # Paper eval metrics.
            "gt_error_xy_mean_m": "task/gt_error_xy_m",
            "gt_error_z_abs_mean_m": "task/gt_error_z_abs_m",
            "precision_hold_steps": "task/precision_hold_steps",
            "near_fail_rate": "task/near_fail_rate",
            "gt_target_z_mean_m": "task/gt_target_z_m",
            "ee_pos_z_mean_m": "task/ee_pos_z_m",
            "yolo_center_x_mean_px": "task/yolo_center_x_px",
            "yolo_center_y_mean_px": "task/yolo_center_y_px",
            "yolo_depth_mean_m": "task/yolo_depth_m",
        }
        for src_key, tag in _TAG_MAP.items():
            if src_key not in rollout_metrics:
                continue
            try:
                self.logger.record(tag, float(rollout_metrics[src_key]))
            except (TypeError, ValueError):
                continue

    def _on_rollout_end(self) -> None:
        # SB3 already prints the train table; this block adds only GT/progress.
        rollout_metrics = self._collect_rollout_metrics()
        self._record_rollout_metrics_to_logger(rollout_metrics)
        self._print_sb3_block(rollout_metrics, self._collect_train_metrics(),
                              skip_train=True)
        self._reset_rollout_totals()

    def _on_step(self) -> bool:
        self._accumulate_gt_stats()
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._t_start
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print("─" * 70)
        print(f"  Training finished, total time: {h:02d}:{m:02d}:{s:02d}")
        print("═" * 70 + "\n")
