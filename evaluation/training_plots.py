"""Paper-standard training-performance plots for the UR10e runs.

Reads the TensorBoard event files written during training (SB3 ``tensorboard_log``)
and renders clean, publication-style figures plus an extracted key-points summary.

Design notes:
- Self-contained TFRecord + protobuf scalar reader (``_read_tfevents_scalars``) so
  plotting works WITHOUT tensorboard/tensorflow installed (e.g. standalone on a
  laptop). Falls back to tensorboard's EventAccumulator only if present and the
  hand-rolled reader yields nothing.
- The task metrics (det_rate, crane_err_mean, gt_error_norm) are logged under the
  ``task/`` namespace by TrainingMonitorCallback._record_rollout_metrics_to_logger
  (training/callbacks/monitor.py), so the plots read the same tags as training.

Usage (standalone, no Isaac needed):
    python -m evaluation.training_plots <run_dir> [--out <out_dir>]

Or from Python:
    from evaluation.training_plots import generate_training_plots
    generate_training_plots(run_dir, out_dir, title=run_tag)
"""

from __future__ import annotations

import glob
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
from evaluation.plotting import apply_paper_style, save_figure


# ─────────────────────────────────────────────────────────────────────────────
# Minimal TFRecord + protobuf scalar reader (no tensorboard dependency).
#
# TFRecord layout per record:  uint64 len | uint32 crc(len) | data | uint32 crc(data)
# Event proto:    step=field2 (varint), summary=field5 (message)
# Summary proto:  value=field1 (repeated message)
# Value proto:    tag=field1 (string), simple_value=field2 (float32)
# ─────────────────────────────────────────────────────────────────────────────
def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _iter_tfrecords(data: bytes):
    i, n = 0, len(data)
    while i + 12 <= n:
        length = struct.unpack_from("<Q", data, i)[0]
        i += 8 + 4  # length + crc(length)
        if i + length + 4 > n:
            break
        yield data[i:i + length]
        i += length + 4  # data + crc(data)


def _fields(buf: bytes):
    """Yield (field number, wire type, payload) of one protobuf message; stops at an unknown type.

    Payload: int for varints (0), bytes for length-delimited (2), raw bytes for fixed64 (1)/fixed32 (5).
    """
    i, n = 0, len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            value, i = _read_varint(buf, i)
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            value = buf[i:i + ln]
            i += ln
        elif wt == 1:
            value = buf[i:i + 8]
            i += 8
        elif wt == 5:
            value = buf[i:i + 4]
            i += 4
        else:
            return
        yield field, wt, value


def _parse_value(buf: bytes) -> Optional[Tuple[str, float]]:
    """Summary.Value: tag (1) + simple_value (2)."""
    tag: Optional[str] = None
    simple: Optional[float] = None
    for field, wt, value in _fields(buf):
        if wt == 2 and field == 1:
            tag = value.decode("utf-8", "replace")
        elif wt == 5 and field == 2:
            simple = struct.unpack("<f", value)[0]
    if tag is not None and simple is not None:
        return tag, simple
    return None


def _parse_summary(buf: bytes) -> List[Tuple[str, float]]:
    """Summary: repeated Value (1)."""
    out: List[Tuple[str, float]] = []
    for field, wt, value in _fields(buf):
        if wt == 2 and field == 1:
            tv = _parse_value(value)
            if tv is not None:
                out.append(tv)
    return out


def _parse_event(record: bytes) -> Tuple[int, List[Tuple[str, float]]]:
    """Event: step (2) + summary (5)."""
    step = 0
    values: List[Tuple[str, float]] = []
    for field, wt, value in _fields(record):
        if wt == 0 and field == 2:
            step = value
        elif wt == 2 and field == 5:
            values.extend(_parse_summary(value))
    return step, values


def _read_tfevents_scalars(path: str) -> Dict[str, List[Tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} from a single tfevents file."""
    out: Dict[str, List[Tuple[int, float]]] = {}
    with open(path, "rb") as f:
        data = f.read()
    for record in _iter_tfrecords(data):
        try:
            step, values = _parse_event(record)
        except (IndexError, struct.error):
            continue
        for tag, value in values:
            if not np.isfinite(value):
                continue
            out.setdefault(tag, []).append((step, value))
    return out


def load_run_scalars(run_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Merge all tfevents under run_dir into {tag: (steps, values)} sorted by step.

    Handles resumes (multiple PPO_N dirs / event files): later writes win on
    duplicate (tag, step).
    """
    event_files = sorted(glob.glob(os.path.join(run_dir, "**", "*tfevents*"), recursive=True))
    merged: Dict[str, Dict[int, float]] = {}
    for ef in event_files:
        try:
            scalars = _read_tfevents_scalars(ef)
        except Exception:
            scalars = {}
        for tag, series in scalars.items():
            d = merged.setdefault(tag, {})
            for step, value in series:
                d[step] = value  # later file overwrites duplicate step

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag, d in merged.items():
        if not d:
            continue
        steps = np.array(sorted(d.keys()), dtype=np.float64)
        vals = np.array([d[int(s)] for s in steps], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def _ema(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out




# (tag, title, y-label, "max"|"min" = which extreme is "best", log_y, color, percent)
# Core comparison panels — always shown (a missing one renders "metric not logged",
# which is a useful diagnostic that training failed to log it).
_PANELS = [
    ("rollout/ep_rew_mean", "Episode Reward", "mean episode reward", "max", False, "#1f77b4", False),
    ("task/success_rate", "Stable Success Rate (GT stay-20)", "stable success", "max", False, "#1fa774", True),
    ("task/gt_reach_rate", "GT Reach Rate", "reached target", "max", False, "#17becf", True),
    ("task/gt_error_norm_m", "Task Error (EE→target)", r"$\|\mathrm{error}\|_2$ (m)", "min", False, "#d62728", False),
    # Paper eval metrics.
    ("task/gt_error_xy_m", "Final Center Error ($\\bar{E}_{xy}$)", r"$\bar{E}_{xy}$ (m)", "min", False, "#e6550d", False),
    ("task/gt_error_z_abs_m", "Final Depth Error ($\\bar{E}_{z}$)", r"$\bar{E}_{z}$ (m)", "min", False, "#756bb1", False),
    ("task/precision_hold_steps", "Precision Hold ($\\bar{H}$)", "held steps (success eps)", "max", False, "#31a354", False),
    ("task/near_fail_rate", "Near-but-Fail Rate (NFR)", "near but fail", "min", False, "#c49c94", True),
    ("task/det_rate", "YOLO Detection Rate", "detection rate", "max", False, "#2ca02c", True),
    ("task/crane_err_mean", "Crane Orientation Error", r"$1-\cos\theta$", "min", True, "#9467bd", False),
    ("train/explained_variance", "Value Fn. Explained Variance", "explained variance", "max", False, "#ff7f0e", False),
]


# (tag, filename, optional standalone y-label override)
_HEADLINES = [
    ('rollout/ep_rew_mean', 'fig_reward', None),
    ('task/gt_error_norm_m', 'fig_task_error', None),
    ('task/det_rate', 'fig_detection_rate', None),
    ('task/success_rate', 'fig_success_rate', 'stable success rate'),
    ('task/gt_error_xy_m', 'fig_center_error', None),
    ('task/gt_error_z_abs_m', 'fig_depth_error', None),
    ('task/precision_hold_steps', 'fig_precision_hold', None),
    ('task/near_fail_rate', 'fig_near_fail_rate', None),
]


def _fmt_step(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _plot_panel(ax, steps, vals, title, ylabel, best, log_y, color, percent) -> Optional[dict]:
    import matplotlib.ticker as mticker

    if steps is None or steps.size == 0:
        ax.set_title(title + "  (no data)")
        ax.text(0.5, 0.5, "metric not logged", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    scale = 100.0 if percent else 1.0
    y = vals * scale
    smooth = _ema(y, alpha=0.3)

    ax.plot(steps, y, color=color, alpha=0.22, linewidth=1.0)
    ax.plot(steps, smooth, color=color, label="EMA")

    # Key points: final + best.
    final_v = y[-1]
    if best == "max":
        idx = int(np.argmax(y))
    else:
        idx = int(np.argmin(y))
    best_v, best_s = y[idx], steps[idx]

    ax.scatter([steps[-1]], [final_v], color=color, s=22, zorder=5)
    ax.scatter([best_s], [best_v], facecolors="none", edgecolors=color, s=55, zorder=5, linewidths=1.4)

    unit = "%" if percent else ""
    ax.annotate(f"final {final_v:.3g}{unit}", xy=(steps[-1], final_v),
                xytext=(-4, 8), textcoords="offset points", ha="right",
                fontsize=8, color=color)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("environment steps")
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: _fmt_step(v)))
    ax.legend(loc="best")

    return {
        "final": float(final_v / scale),
        "best": float(best_v / scale),
        "best_step": float(best_s),
        "final_step": float(steps[-1]),
    }


def generate_training_plots(
    run_dir: str,
    out_dir: str,
    title: Optional[str] = None,
) -> dict:
    """Render the training dashboard + headline figures + key-points summary.

    Returns a dict of extracted key points. Writes PNG + PDF + a text summary
    into out_dir. Never raises on missing metrics — panels degrade gracefully.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
    os.makedirs(out_dir, exist_ok=True)

    scalars = load_run_scalars(run_dir)
    if not scalars:
        print(f"[PLOTS][WARN] no tfevents scalars found under {run_dir}", flush=True)
        return {}

    suptitle = title or os.path.basename(os.path.normpath(run_dir))

    # ── Dashboard (dynamic grid) ──
    panels = list(_PANELS)
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.1 * nrows))
    axes_list = list(np.atleast_1d(axes).flatten())
    keypoints: dict = {}
    for ax, (tag, t, yl, best, logy, color, percent) in zip(axes_list, panels):
        series = scalars.get(tag)
        steps, vals = series if series is not None else (None, None)
        kp = _plot_panel(ax, steps, vals, t, yl, best, logy, color, percent)
        if kp is not None:
            keypoints[tag] = kp
    for ax in axes_list[len(panels):]:  # hide unused cells
        ax.axis("off")

    fig.suptitle(f"Training Performance — {suptitle}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dash_png, dash_pdf = save_figure(fig, out_dir, "training_overview")
    print(f"[PLOTS] dashboard -> {dash_png}", flush=True)

    # ── Headline single figures (reward, task error, detection rate) ──
    panel_by_tag = {panel[0]: panel for panel in panels}
    for tag, fname, ylabel_override in _HEADLINES:
        _, t, yl, best, logy, color, percent = panel_by_tag[tag]
        if ylabel_override is not None:
            yl = ylabel_override
        series = scalars.get(tag)
        if series is None:
            continue
        f1, ax1 = plt.subplots(figsize=(6.2, 4.0))
        _plot_panel(ax1, series[0], series[1], t, yl, best, logy, color, percent)
        f1.tight_layout()
        save_figure(f1, out_dir, fname)
    print(f"[PLOTS] headline figures -> {out_dir}", flush=True)

    # ── Key points text ──
    _write_keypoints(os.path.join(out_dir, "training_keypoints.txt"), suptitle, scalars, keypoints)
    return keypoints


_KEYPOINT_LABELS = [
    ("rollout/ep_rew_mean", "Episode reward", "max", ""),
    ("task/success_rate", "Stable success rate (GT stay-20)", "max", "(frac)"),
    ("task/gt_success_rate", "GT stable success rate", "max", "(frac)"),
    ("task/gt_reach_rate", "GT reach rate", "max", "(frac)"),
    ("task/gt_error_norm_m", "Task error |EE-target|", "min", "m"),
    ("task/gt_error_xy_m", "Final center error (Exy)", "min", "m"),
    ("task/gt_error_z_abs_m", "Final depth error (Ez)", "min", "m"),
    ("task/precision_hold_steps", "Precision hold (H)", "max", "steps"),
    ("task/near_fail_rate", "Near-but-fail rate (NFR)", "min", "(frac)"),
    ("task/det_rate", "Detection rate", "max", "(frac)"),
    ("task/crane_err_mean", "Crane error (1-cos)", "min", ""),
    ("task/gt_error_z_m", "Z error (signed)", "min_abs", "m"),
    ("train/explained_variance", "Explained variance", "max", ""),
    ("train/approx_kl", "Approx KL", "min", ""),
]


def _write_keypoints(path: str, title: str, scalars: dict, keypoints: dict) -> None:
    lines = []
    lines.append(f"Training key points — {title}")
    lines.append("=" * 64)
    # Determine total steps from any series.
    total_steps = 0.0
    for s, _v in scalars.values():
        if s.size:
            total_steps = max(total_steps, float(s[-1]))
    lines.append(f"Total environment steps: {_fmt_step(total_steps)} ({int(total_steps):,})")
    lines.append("")
    lines.append(f"{'metric':<28}{'final':>12}{'best':>12}{'best@step':>14}")
    lines.append("-" * 66)
    for tag, label, best, unit in _KEYPOINT_LABELS:
        series = scalars.get(tag)
        if series is None or series[0].size == 0:
            continue
        steps, vals = series
        final_v = float(vals[-1])
        if best == "max":
            idx = int(np.argmax(vals))
        elif best == "min_abs":
            idx = int(np.argmin(np.abs(vals)))
        else:
            idx = int(np.argmin(vals))
        best_v = float(vals[idx])
        u = f" {unit}" if unit else ""
        lines.append(f"{label:<28}{final_v:>12.4g}{best_v:>12.4g}{_fmt_step(steps[idx]):>14}{u}")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(f"[PLOTS] key points -> {path}", flush=True)
    print(text, flush=True)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate paper-standard training plots from tfevents.")
    ap.add_argument("run_dir", help="Run directory (contains PPO_*/events.out.tfevents.*)")
    ap.add_argument("--out", default=None, help="Output dir (default: <run_dir>/plots)")
    ap.add_argument("--title", default=None, help="Figure title tag")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run_dir, "plots")
    generate_training_plots(args.run_dir, out, title=args.title)


if __name__ == "__main__":
    _main()
