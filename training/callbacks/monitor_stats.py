"""Stat-accumulation mixin for TrainingMonitorCallback.

Holds the GT / YOLO / crane per-env accumulators harvested at
episode end. Split out of monitor.py so the callback's lifecycle + printing code
stays under the line budget. These methods rely on attributes and
`_get_isaaclab_env()` defined on TrainingMonitorCallback (resolved via MRO).
"""

import torch
from isaaclab.utils.math import matrix_from_quat

from rl_env.env import (
    APPROACH_AXIS,
    CRANE_TARGET_AXIS_W,
    EE_LINK_NAME,
    GT_SUCCESS_XY_TOL_M,
    GT_SUCCESS_Z_TOL_M,
    SCRIPTED_STAY_STEPS_REQUIRED,
)
from rl_env.env_ids import find_body_id
from rl_env.rewards.error_sources import gt_task_target_pos_w
from rl_env.success_metrics import advance_stable_success, per_axis_in_band
from config import (
    YOLO_CACHE_NORM_DEPTH_M,
    YOLO_CACHE_NORM_PX,
)


class _MonitorStatsMixin:
    """GT/YOLO/crane accumulators for TrainingMonitorCallback."""

    def _get_yolo_debug(self):
        cache = getattr(self._isaaclab_env, "_yolo_detection_cache", None)
        if not isinstance(cache, dict):
            return None
        has_det = cache.get("has_det", None)
        centers = cache.get("center_norm", None)
        depths = cache.get("depth_norm", None)
        if has_det is None or centers is None or depths is None:
            return None
        try:
            return (
                has_det.detach().float(),
                centers.detach().float(),
                depths.detach().float().squeeze(-1),
            )
        except Exception:
            return None

    def _store_last_yolo_debug(self, yolo_debug) -> None:
        if yolo_debug is None:
            self._last_yolo_has_det = None
            self._last_yolo_centers = None
            self._last_yolo_depths = None
            return
        has_det, centers, depths = yolo_debug
        self._last_yolo_has_det = has_det.detach().clone()
        self._last_yolo_centers = centers.detach().clone()
        self._last_yolo_depths = depths.detach().clone()

    def _accumulate_last_yolo_debug(self, mask: torch.Tensor) -> None:
        has_det = getattr(self, "_last_yolo_has_det", None)
        centers = getattr(self, "_last_yolo_centers", None)
        depths = getattr(self, "_last_yolo_depths", None)
        if has_det is None or centers is None or depths is None:
            return
        if has_det.shape[0] != mask.shape[0] or centers.shape[0] != mask.shape[0] or depths.shape[0] != mask.shape[0]:
            return
        mask_bool = mask.to(device=has_det.device, dtype=torch.bool)
        self._yolo_total_env_steps += int(mask_bool.sum().item())
        det_mask = (has_det > 0.5) & mask_bool
        det_count = int(det_mask.sum().item())
        if det_count <= 0:
            return
        self._yolo_center_x_sum += float(centers[det_mask, 0].sum().item()) * YOLO_CACHE_NORM_PX
        self._yolo_center_y_sum += float(centers[det_mask, 1].sum().item()) * YOLO_CACHE_NORM_PX
        self._yolo_depth_sum += float(depths[det_mask].sum().item()) * YOLO_CACHE_NORM_DEPTH_M
        self._yolo_metric_count += det_count

    def _ee_id(self, robot) -> int | None:
        """EE body index, resolved once per training run (shared by crane and GT stats)."""
        if self._ee_body_id is None:
            self._ee_body_id = find_body_id(robot, EE_LINK_NAME)
        return self._ee_body_id

    def _get_crane_error_debug(self) -> torch.Tensor | None:
        if self._isaaclab_env is None:
            return None
        try:
            robot = self._isaaclab_env.scene["robot"]
            body_id = self._ee_id(robot)
            if body_id is None:
                return None

            quat_w = robot.data.body_quat_w[:, body_id]
            rot_mat = matrix_from_quat(quat_w)
            axis_w = rot_mat[:, :, int(APPROACH_AXIS)]
            target = torch.tensor(CRANE_TARGET_AXIS_W, device=axis_w.device, dtype=axis_w.dtype)
            target = target / torch.clamp(torch.linalg.norm(target), min=1e-6)
            cos_sim = torch.sum(axis_w * target.unsqueeze(0), dim=1)
            return torch.clamp(1.0 - cos_sim, 0.0, 2.0).detach().float()
        except Exception:
            return None

    def _store_last_crane_error(self, crane_err: torch.Tensor | None) -> None:
        self._last_crane_err = None if crane_err is None else crane_err.detach().clone()

    def _accumulate_last_crane_error(self, mask: torch.Tensor) -> None:
        crane_err = getattr(self, "_last_crane_err", None)
        if crane_err is None or crane_err.shape[0] != mask.shape[0]:
            return
        mask = mask.to(device=crane_err.device, dtype=crane_err.dtype)
        self._crane_err_sum += float((crane_err * mask).sum().item())
        self._crane_err_count += int(mask.sum().item())

    def _accumulate_gt_stats(self) -> None:
        """Record GT target/EE/error at episode END (final step) via per-env buffer."""
        if self._isaaclab_env is None:
            try:
                self._isaaclab_env = self._get_isaaclab_env()
            except Exception:
                return
        try:
            from isaaclab.assets import Articulation

            robot: Articulation = self._isaaclab_env.scene["robot"]

            ee_id = self._ee_id(robot)
            if ee_id is None:
                return

            ee_pos_w = robot.data.body_pos_w[:, ee_id]
            target_pos_w = gt_task_target_pos_w(self._isaaclab_env)
            gt_error_w = target_pos_w - ee_pos_w
            # GT at-target THIS step (true reach, GT regardless of variant) — for a fair
            # cross-ablation success metric independent of the (GT vs vision) reward source.
            gt_at = per_axis_in_band(gt_error_w, GT_SUCCESS_XY_TOL_M, GT_SUCCESS_Z_TOL_M)
            # "Near" = within a relaxed (2×) tolerance — used for Near-but-Fail Rate (NFR):
            # episodes that approached the target but never satisfied the strict success tol.
            _NEAR_FACTOR = 2.0
            near_now = per_axis_in_band(
                gt_error_w,
                float(GT_SUCCESS_XY_TOL_M) * _NEAR_FACTOR,
                float(GT_SUCCESS_Z_TOL_M) * _NEAR_FACTOR,
            )

            # ── Per-env buffer for last-step stats ──
            num_envs = ee_pos_w.shape[0]
            yolo_debug = self._get_yolo_debug()
            crane_debug = self._get_crane_error_debug()
            ep_len_buf = getattr(self._isaaclab_env, "episode_length_buf", None)
            ep_len = None
            if ep_len_buf is not None:
                ep_len = ep_len_buf.detach().clone() if torch.is_tensor(ep_len_buf) else torch.as_tensor(ep_len_buf, device=ee_pos_w.device)

            first_buffer = (
                not hasattr(self, "_last_ee")
                or self._last_ee is None
                or self._last_ee.shape[0] != num_envs
            )
            if first_buffer:
                self._gt_ever_at_target = gt_at.detach().clone()
                self._hold_count = gt_at.detach().to(torch.long).clone()
                self._gt_stay_streak = gt_at.detach().to(torch.long).clone()
                self._gt_stable_success = self._gt_stay_streak >= int(SCRIPTED_STAY_STEPS_REQUIRED)
                self._gt_ever_near = near_now.detach().clone()
                self._last_ee = ee_pos_w.detach().clone()
                self._last_target = target_pos_w.detach().clone()
                self._last_error = gt_error_w.detach().clone()
                self._store_last_yolo_debug(yolo_debug)
                self._store_last_crane_error(crane_debug)
                self._last_episode_length_buf = ep_len
            else:
                # Episode reset is a wrap/drop in episode_length_buf. Checking <= 1
                # double-counts the first step of the new episode as a fake final step.
                last_ep_len = getattr(self, "_last_episode_length_buf", None)
                if ep_len is not None and last_ep_len is not None and last_ep_len.shape[0] == num_envs:
                    just_reset = ep_len < last_ep_len
                    if torch.any(just_reset):
                        mask = just_reset.to(dtype=torch.float32)
                        self._gt_target_x_sum += float((self._last_target[:, 0] * mask).sum().item())
                        self._gt_target_y_sum += float((self._last_target[:, 1] * mask).sum().item())
                        self._gt_target_z_sum += float((self._last_target[:, 2] * mask).sum().item())
                        self._ee_pos_x_sum += float((self._last_ee[:, 0] * mask).sum().item())
                        self._ee_pos_y_sum += float((self._last_ee[:, 1] * mask).sum().item())
                        self._ee_pos_z_sum += float((self._last_ee[:, 2] * mask).sum().item())
                        self._gt_error_x_sum += float((self._last_error[:, 0] * mask).sum().item())
                        self._gt_error_y_sum += float((self._last_error[:, 1] * mask).sum().item())
                        self._gt_error_z_sum += float((self._last_error[:, 2] * mask).sum().item())
                        _err_norm = torch.sqrt(
                            (self._last_error[:, 0]**2 + self._last_error[:, 1]**2 + self._last_error[:, 2]**2).clamp(min=0.0)
                        ) * mask
                        self._gt_error_norm_sum += float(_err_norm.sum().item())
                        # Final Center Error (xy) and Final Depth Error (|z|) at episode end.
                        _err_xy = torch.sqrt(
                            (self._last_error[:, 0]**2 + self._last_error[:, 1]**2).clamp(min=0.0)
                        ) * mask
                        self._gt_error_xy_sum += float(_err_xy.sum().item())
                        self._gt_error_z_abs_sum += float((self._last_error[:, 2].abs() * mask).sum().item())
                        self._gt_metric_count += int(just_reset.sum().item())
                        _gt_flag = getattr(self, "_gt_ever_at_target", None)
                        _stable_flag = getattr(self, "_gt_stable_success", None)
                        if (
                            _gt_flag is not None
                            and _stable_flag is not None
                            and _gt_flag.shape[0] == num_envs
                            and _stable_flag.shape[0] == num_envs
                        ):
                            _reach_reset = _gt_flag & just_reset
                            _succ_reset = _stable_flag & just_reset
                            self._gt_reach_count += int(_reach_reset.sum().item())
                            self._gt_success_count += int(_succ_reset.sum().item())
                            # Precision Hold: mean #held-steps over SUCCESS episodes only.
                            _hc = getattr(self, "_hold_count", None)
                            if _hc is not None and _hc.shape[0] == num_envs:
                                self._hold_steps_sum += float(_hc[_succ_reset].sum().item())
                            self._hold_success_episodes += int(_succ_reset.sum().item())
                            # Near-but-Fail: ever within NEAR tol AND not a success this episode.
                            _nf = getattr(self, "_gt_ever_near", None)
                            if _nf is not None and _nf.shape[0] == num_envs:
                                self._near_fail_count += int(((_nf & ~_stable_flag) & just_reset).sum().item())
                        self._accumulate_last_yolo_debug(mask)
                        self._accumulate_last_crane_error(mask)

                # Store current values for next step's comparison
                self._last_ee = ee_pos_w.detach().clone()
                self._last_target = target_pos_w.detach().clone()
                self._last_error = gt_error_w.detach().clone()
                self._store_last_yolo_debug(yolo_debug)
                self._store_last_crane_error(crane_debug)
                self._last_episode_length_buf = ep_len

                # Update the GT ever-at-target flag: reset envs start fresh at this step's
                # at-target; continuing envs OR-accumulate. Recorded above on episode end.
                _prev_flag = getattr(self, "_gt_ever_at_target", None)
                if _prev_flag is None or _prev_flag.shape[0] != num_envs:
                    self._gt_ever_at_target = gt_at.detach().clone()
                    self._hold_count = gt_at.detach().to(torch.long).clone()
                    self._gt_stay_streak = gt_at.detach().to(torch.long).clone()
                    self._gt_stable_success = self._gt_stay_streak >= int(SCRIPTED_STAY_STEPS_REQUIRED)
                    self._gt_ever_near = near_now.detach().clone()
                else:
                    if ep_len is not None and last_ep_len is not None and last_ep_len.shape[0] == num_envs:
                        _jr = ep_len < last_ep_len
                    else:
                        _jr = torch.zeros_like(gt_at)
                    self._gt_ever_at_target = torch.where(_jr, torch.zeros_like(gt_at), _prev_flag | gt_at)
                    # Precision-hold step counter + near flag: increment/OR over the episode,
                    # fresh at zero on reset; reset observations are episode_length=0,
                    # not completed control steps in the new episode.
                    _gt_at_long = gt_at.to(torch.long)
                    _hc_prev = getattr(self, "_hold_count", None)
                    if _hc_prev is None or _hc_prev.shape[0] != num_envs:
                        self._hold_count = _gt_at_long.clone()
                    else:
                        self._hold_count = torch.where(_jr, torch.zeros_like(_gt_at_long), _hc_prev + _gt_at_long)
                    _streak_prev = getattr(self, "_gt_stay_streak", None)
                    _stable_prev = getattr(self, "_gt_stable_success", None)
                    if _streak_prev is None or _streak_prev.shape[0] != num_envs:
                        _streak_prev = torch.zeros_like(_gt_at_long)
                    if _stable_prev is None or _stable_prev.shape[0] != num_envs:
                        _stable_prev = torch.zeros_like(gt_at)
                    self._gt_stay_streak, self._gt_stable_success = advance_stable_success(
                        _streak_prev, _stable_prev, gt_at, _jr, SCRIPTED_STAY_STEPS_REQUIRED
                    )
                    _nr_prev = getattr(self, "_gt_ever_near", None)
                    if _nr_prev is None or _nr_prev.shape[0] != num_envs:
                        self._gt_ever_near = near_now.detach().clone()
                    else:
                        self._gt_ever_near = torch.where(_jr, torch.zeros_like(near_now), _nr_prev | near_now)
        except Exception as _exc:
            # This used to swallow ALL errors silently — a single bug here zeroes every
            # rollout metric (shown as "-"). Surface the first occurrence so it isn't hidden.
            if not getattr(self, "_gt_stats_warned", False):
                import traceback
                print(f"[MONITOR] _accumulate_gt_stats disabled after error: {_exc}", flush=True)
                traceback.print_exc()
                self._gt_stats_warned = True
            return
