"""Printing mixin for TrainingMonitorCallback (console table + float formatting).

Split out of monitor.py to keep it under the line budget. Methods rely on
attributes set on TrainingMonitorCallback (resolved via MRO).
"""

import time

import numpy as np


class _MonitorPrintMixin:
    """Console rollout-table rendering for TrainingMonitorCallback."""

    def _format_float(self, value) -> str:
        if isinstance(value, (int, np.integer)):
            return f"{int(value)}"
        if isinstance(value, float):
            # Always non-scientific for console readability.
            if np.isnan(value):
                return "nan"
            if np.isinf(value):
                return "inf" if value > 0 else "-inf"
            text = f"{value:,.6f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text
        return str(value)

    def _print_sb3_block(self, rollout_metrics: dict[str, object], train_metrics: dict[str, object],
                         skip_train: bool = False) -> None:
        self._rollout_count += 1
        now = time.time()
        elapsed_seconds = max(now - self._t_start, 1e-6)
        current_steps = int(self.model.num_timesteps)
        # Compute progress relative to steps planned for this run only.
        start_steps = int(getattr(self, "_start_steps", 0))
        total_steps = int(self._total_steps) if self._total_steps is not None else 0
        run_total = max(total_steps - start_steps, 0)
        run_progress = max(current_steps - start_steps, 0)
        remaining_steps = max(run_total - run_progress, 0)

        # ETA uses a smoothed recent collection rate. Cumulative rate is misleading
        # after slow startup, and especially after resume because checkpoint steps
        # are historical while elapsed_seconds is only this run.
        dt_recent = max(now - float(getattr(self, "_eta_t_last", self._t_start)), 1e-6)
        d_steps_recent = max(current_steps - int(getattr(self, "_eta_steps_last", current_steps)), 0)
        recent_rate = d_steps_recent / dt_recent if d_steps_recent > 0 else 0.0
        if recent_rate > 0.0:
            if self._eta_rate_ema is None:
                self._eta_rate_ema = recent_rate
            else:
                self._eta_rate_ema = 0.35 * recent_rate + 0.65 * self._eta_rate_ema
            self._eta_t_last = now
            self._eta_steps_last = current_steps
        eta_rate = self._eta_rate_ema or (run_progress / elapsed_seconds if run_progress > 0 else 0.0)
        eta_seconds = int(remaining_steps / max(eta_rate, 1e-6)) if run_total > 0 and run_progress > 0 else 0
        progress_pct = 0.0
        if run_total > 0:
            progress_pct = 100.0 * float(run_progress) / float(run_total)

        def _fmt_hms(seconds: int) -> str:
            seconds = max(int(seconds), 0)
            hours, remainder = divmod(seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        _SEP = "-" * 62
        print(_SEP)
        if not skip_train:
            print(f"| {'rollout/':<60}|")
            ep_len_text = self._format_float(rollout_metrics.get("ep_len_mean")) if "ep_len_mean" in rollout_metrics else "(menunggu ep)"
            ep_rew_text = self._format_float(rollout_metrics.get("ep_rew_mean")) if "ep_rew_mean" in rollout_metrics else "(menunggu ep)"
            print(f"|    {'ep_len_mean':<22}| {ep_len_text:<32}|")
            print(f"|    {'ep_rew_mean':<22}| {ep_rew_text:<32}|")
        # Final-episode YOLO detection rate + bbox center/depth (raw px/m)
        det_rate_text = f"{float(rollout_metrics['det_rate']):.1%}" if "det_rate" in rollout_metrics else "-"
        yolo_center_text = "-"
        yolo_depth_text = "-"
        if "yolo_center_x_mean_px" in rollout_metrics and "yolo_center_y_mean_px" in rollout_metrics:
            yolo_center_text = (
                f"({float(rollout_metrics['yolo_center_x_mean_px']):.1f}, "
                f"{float(rollout_metrics['yolo_center_y_mean_px']):.1f}) px"
            )
        if "yolo_depth_mean_m" in rollout_metrics:
            yolo_depth_text = f"{float(rollout_metrics['yolo_depth_mean_m']):.4f} m"
        print(f"|    {'det_rate_final':<22}| {det_rate_text:<32}|")
        print(f"|    {'yolo_center_final':<22}| {yolo_center_text:<32}|")
        print(f"|    {'yolo_depth_final':<22}| {yolo_depth_text:<32}|")
        # GT position/error summary (meters, world frame)
        gt_pos_text = "-"
        ee_pos_text = "-"
        gt_err_text = "-"
        gt_err_norm_text = "-"
        if "gt_target_x_mean_m" in rollout_metrics:
            gt_pos_text = (
                f"({float(rollout_metrics['gt_target_x_mean_m']):.4f}, "
                f"{float(rollout_metrics['gt_target_y_mean_m']):.4f}, "
                f"{float(rollout_metrics['gt_target_z_mean_m']):.4f}) m"
            )
        if "ee_pos_x_mean_m" in rollout_metrics:
            ee_pos_text = (
                f"({float(rollout_metrics['ee_pos_x_mean_m']):.4f}, "
                f"{float(rollout_metrics['ee_pos_y_mean_m']):.4f}, "
                f"{float(rollout_metrics['ee_pos_z_mean_m']):.4f}) m"
            )
        if "gt_error_x_mean_m" in rollout_metrics:
            gt_err_text = (
                f"({float(rollout_metrics['gt_error_x_mean_m']):+.4f}, "
                f"{float(rollout_metrics['gt_error_y_mean_m']):+.4f}, "
                f"{float(rollout_metrics['gt_error_z_mean_m']):+.4f}) m"
            )
        if "gt_error_norm_mean_m" in rollout_metrics:
            gt_err_norm_text = f"{float(rollout_metrics['gt_error_norm_mean_m']):.4f} m"
        crane_err_text = self._format_float(float(rollout_metrics['crane_err_mean'])) if "crane_err_mean" in rollout_metrics else "-"
        print(f"|    {'crane_err_final':<22}| {crane_err_text:<32}|")
        print(f"|    {'gt_target_pos (xyz m)':<22}| {gt_pos_text:<32}|")
        print(f"|    {'ee_pos (xyz m)':<22}| {ee_pos_text:<32}|")
        print(f"|    {'gt_error (xyz m)':<22}| {gt_err_text:<32}|")
        print(f"|    {'gt_error_norm (m)':<22}| {gt_err_norm_text:<32}|")
        if "gt_success_rate" in rollout_metrics:
            print(f"|    {'gt_success_rate':<22}| {float(rollout_metrics['gt_success_rate'])*100:.1f}% (GT stay-20){'':<13}|")
        if "gt_reach_rate" in rollout_metrics:
            print(f"|    {'gt_reach_rate':<22}| {float(rollout_metrics['gt_reach_rate'])*100:.1f}% (>=1 GT in-band){'':<9}|")
        # Paper eval metrics.
        if "gt_error_xy_mean_m" in rollout_metrics:
            _exy = float(rollout_metrics['gt_error_xy_mean_m'])
            _ez = float(rollout_metrics.get('gt_error_z_abs_mean_m', 0.0))
            print(f"|    {'final_err (xy,z) m':<22}| ({_exy:.4f}, {_ez:.4f}){'':<16}|")
        if "near_fail_rate" in rollout_metrics:
            print(f"|    {'near_but_fail_rate':<22}| {float(rollout_metrics['near_fail_rate'])*100:.1f}%{'':<26}|")
        if "precision_hold_steps" in rollout_metrics:
            print(f"|    {'precision_hold':<22}| {float(rollout_metrics['precision_hold_steps']):.1f} steps{'':<19}|")
        if run_total > 0:
            progress_text = f"{run_progress}/{run_total} ({progress_pct:.1f}%)"
            print(f"|    {'step_progress':<22}| {progress_text:<32}|")
            print(f"|    {'time_remaining':<22}| {_fmt_hms(eta_seconds):<32}|")
        # Train metrics: neat, model-specific block (one metric per line).
        # PPO: SB3 built-in already prints train table → skip_train=True avoids duplication.
        if not skip_train:
            ordered_keys = self._train_metric_keys()
            if ordered_keys:
                print(f"| {f'train/{self._model_type}':<60}|")
                any_metric = False
                for key in ordered_keys:
                    if key not in train_metrics:
                        continue
                    any_metric = True
                    print(f"|    {key:<22}| {self._format_float(train_metrics[key]):<32}|")
                if not any_metric:
                    print(f"|    {'(menunggu update)':<22}| {'-':<32}|")
        print(_SEP)

    def _train_metric_keys(self) -> list:
        return [
            "policy_gradient_loss", "value_loss", "entropy_loss",
            "approx_kl", "clip_fraction", "explained_variance",
            "learning_rate", "n_updates", "std",
        ]
