"""
Render INFERENCE validation figures — one file per metric, INFERENCE-ONLY, PAPER STYLE.

Standalone (numpy + matplotlib only — NO Isaac): reads
    <out_dir>/inference_metrics.json   (inference aggregates + tolerances)
    <out_dir>/inference_records.npz    (per-episode raw arrays, mm)
and writes ONE figure per metric (PNG @300dpi + PDF), styled to MATCH the training_plots
figures (serif, no top/right spines, muted per-metric colours, same math labels) so each
can be placed head-to-head next to its training counterpart. No training data is drawn.

    fig_center_error   E_xy density        (colour matches training task/gt_error_xy_m)
    fig_depth_error    E_z density
    fig_task_error     norm density
    fig_precision_hold hold-steps density
    fig_center_error_cdf  E_xy ECDF (+ % within tol)
    fig_bias_scatter   per-episode (ex,ey) bias vs scatter
    fig_success_rate / fig_near_fail_rate   single-value stat cards

Re-render anytime without re-running the sim:
    python3 plot_validation.py <out_dir>
"""
import json
import os
import sys

import numpy as np
from evaluation.plotting import apply_paper_style, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colours mirror the matching training_plots panels (evaluation/training_plots.py::_PANELS).
COL = {
    "exy": "#e6550d", "ez": "#756bb1", "norm": "#d62728",
    "hold": "#31a354", "success": "#1fa774", "near": "#c49c94",
    "dact": "#3182bd", "stay20": "#8c6d31",
}
C_TOL = "#444444"
C_MEAN = "#111111"


def _apply_paper_style():
    apply_paper_style({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                       "lines.linewidth": 1.8, "legend.fontsize": 9.5})


def _save(fig, out_dir, fname):
    save_figure(fig, out_dir, fname)


def _kde(x, grid, bw):
    d = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(1) / (x.size * bw * np.sqrt(2 * np.pi))


def _fmt(x, unit):
    s = f"{x:.3g}"
    return f"{s} {unit}".strip()


def _density(out_dir, fname, vals, tol, xlabel, title, n_ep, color, unit="mm"):
    vals = np.asarray(vals, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if vals.size:
        # Data-driven x-range so tight distributions (e.g. E_z ~ 0) stay readable.
        # Only fold the tolerance into the range/plot when it is actually on-scale —
        # otherwise a huge tol (Z=8mm) would flatten a sub-mm distribution into a spike.
        xhi = max(float(np.percentile(vals, 99.5)), float(vals.mean()) * 1.3, 1e-6)
        show_tol = bool(tol) and tol <= xhi * 1.15
        if show_tol:
            xhi = max(xhi, tol * 1.1)
        v = np.clip(vals, 0, xhi)
        _counts, edges, _ = ax.hist(v, bins=36, range=(0, xhi), color=color, alpha=0.30,
                                    edgecolor="white", linewidth=0.3)
        # KDE overlay (Silverman bw), scaled to the histogram counts.
        std = max(float(vals.std()), 1e-9)
        bw = max(1.06 * std * vals.size ** (-1 / 5), xhi / 256)
        grid = np.linspace(0, xhi, 256)
        binw = edges[1] - edges[0]
        ax.plot(grid, _kde(v, grid, bw) * vals.size * binw, color=color, lw=2.0)
        mean, med = float(vals.mean()), float(np.median(vals))
        ax.axvline(mean, color=C_MEAN, ls="-", lw=1.3, label=f"mean {_fmt(mean, unit)}")
        ax.axvline(med, color=C_MEAN, ls=":", lw=1.3, label=f"median {_fmt(med, unit)}")
        if show_tol:
            ax.axvline(tol, color=C_TOL, ls="--", lw=1.5, label=f"tol {_fmt(tol, unit)}")
        ax.set_xlim(0, xhi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"episodes  (n = {n_ep})")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_dir, fname)


def _small_bar(out_dir, fname, value, title, color):
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    v = float(value) if value is not None and value == value else 0.0
    ax.bar([0], [v], width=0.42, color=color, alpha=0.9)
    ax.annotate(f"{v:.3f}", (0, v), ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=C_MEAN, xytext=(0, 3), textcoords="offset points")
    ax.set_xticks([]); ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, fname)


def render_all(out_dir: str):
    _apply_paper_style()
    with open(os.path.join(out_dir, "inference_metrics.json")) as f:
        M = json.load(f)
    rec = np.load(os.path.join(out_dir, "inference_records.npz"))
    exy = rec["exy_mm"]; ez = rec["ez_mm"]; norm = rec["norm_mm"]
    ex = rec["ex_mm"]; ey = rec["ey_mm"]
    hold = rec["hold"] if "hold" in rec.files else np.array([])
    dact = rec["dact_rms_hold"] if "dact_rms_hold" in rec.files else np.array([])
    steps20 = rec["steps_to_stay20"] if "steps_to_stay20" in rec.files else np.array([])

    inf = M["inference_deterministic"]
    tol_xy = float(M["tolerances_m"]["xy"]) * 1000.0
    tol_z = float(M["tolerances_m"]["z"]) * 1000.0
    n_ep = int(M["episodes_completed"])
    written = []

    _density(out_dir, "fig_center_error", exy, tol_xy, r"$\bar{E}_{xy}$  center error (mm)",
             r"Inference: Final Center Error ($\bar{E}_{xy}$)", n_ep, COL["exy"])
    written.append("fig_center_error")
    _density(out_dir, "fig_depth_error", ez, tol_z, r"$\bar{E}_{z}$  depth error (mm)",
             r"Inference: Final Depth Error ($\bar{E}_{z}$)", n_ep, COL["ez"])
    written.append("fig_depth_error")
    _density(out_dir, "fig_task_error", norm, tol_xy, r"$\|\mathrm{error}\|_2$  (mm)",
             r"Inference: Task Error (EE$\rightarrow$target)", n_ep, COL["norm"])
    written.append("fig_task_error")
    _density(out_dir, "fig_precision_hold", hold, None, r"held steps  ($\bar{H}$)",
             r"Inference: Precision Hold ($\bar{H}$)", n_ep, COL["hold"], unit="")
    written.append("fig_precision_hold")

    # ── Motion metrics: hold-phase action fluctuation + control steps to reach 20-step stay.
    # Episodes that never entered the band have no hold phase → len(dact) ≤ n_ep.
    stay_req = int(inf.get("stay_steps_required", 20))
    _density(out_dir, "fig_action_fluctuation_hold", dact, None, r"$|\Delta a|$ RMS  (hold)",
             r"Inference: Action Fluctuation (hold)", len(dact), COL["dact"], unit="")
    written.append("fig_action_fluctuation_hold")
    _density(out_dir, "fig_steps_to_stay20", steps20, None,
             rf"control steps to reach {stay_req}-step stay",
             rf"Inference: Steps to Settle ({stay_req}-step stay)",
             len(steps20), COL["stay20"], unit="")
    written.append("fig_steps_to_stay20")

    # ── ECDF of E_xy ──
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs = np.sort(exy); ys = np.arange(1, xs.size + 1) / max(xs.size, 1)
    xhi = max(float(np.percentile(exy, 99.5)) if exy.size else tol_xy * 2, tol_xy * 1.4)
    ax.plot(xs, ys, color=COL["exy"], lw=2.0)
    frac_in = float((exy <= tol_xy).mean()) if exy.size else float("nan")
    ax.axvline(tol_xy, color=C_TOL, ls="--", lw=1.5, label=f"tol {tol_xy:.1f} mm")
    ax.axhline(frac_in, color=C_TOL, ls=":", lw=1.0)
    ax.text(tol_xy, frac_in, f"  {frac_in*100:.1f}% within {tol_xy:.0f} mm", va="bottom",
            ha="left", fontsize=10, color=C_TOL, fontweight="bold")
    ax.set_xlim(0, xhi); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\bar{E}_{xy}$  center error (mm)"); ax.set_ylabel("cumulative fraction")
    ax.set_title(r"Inference: $\bar{E}_{xy}$ empirical CDF")
    ax.legend(loc="lower right"); fig.tight_layout()
    _save(fig, out_dir, "fig_center_error_cdf"); written.append("fig_center_error_cdf")

    # ── bias vs scatter ──
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(ex, ey, s=7, alpha=0.22, color=COL["exy"], edgecolors="none")
    th = np.linspace(0, 2 * np.pi, 256)
    ax.plot(tol_xy * np.cos(th), tol_xy * np.sin(th), color=C_TOL, ls="--", lw=1.5,
            label=f"tol {tol_xy:.1f} mm")
    mx, my = float(np.mean(ex)), float(np.mean(ey))
    ax.scatter([mx], [my], s=150, color=C_MEAN, marker="X", zorder=5,
               label=f"mean ({mx:+.2f}, {my:+.2f}) mm")
    ax.axhline(0, color="#bbb", lw=0.7); ax.axvline(0, color="#bbb", lw=0.7)
    lim = max(xhi, tol_xy * 1.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"signed $e_x$ (mm)"); ax.set_ylabel(r"signed $e_y$ (mm)")
    ax.set_title("Inference: per-episode error (bias vs scatter)")
    ax.legend(loc="upper right"); fig.tight_layout()
    _save(fig, out_dir, "fig_bias_scatter"); written.append("fig_bias_scatter")

    # ── single-value metrics (simple small bar) ──
    _small_bar(out_dir, "fig_success_rate", inf.get("success_rate"),
               "Inference: Stable Success Rate (stay-20)", COL["success"])
    written.append("fig_success_rate")
    _small_bar(out_dir, "fig_near_fail_rate", inf.get("near_fail_rate"),
               "Inference: Near-but-Fail Rate (NFR)", COL["near"])
    written.append("fig_near_fail_rate")

    return [w + ".png" for w in written]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 plot_validation.py <out_dir>", file=sys.stderr)
        sys.exit(1)
    figs = render_all(sys.argv[1])
    print(f"[PLOT] wrote {len(figs)} figures (PNG+PDF) → {sys.argv[1]}")
    for fn in figs:
        print(f"  {fn}")
