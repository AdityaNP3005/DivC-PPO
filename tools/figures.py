"""Paper Figs. 3-5 from the outputs of tools/reproduce.py (host-side, numpy + matplotlib).

    python3 tools/figures.py [results]      # writes <results>/fig3_eval_error.pdf, fig4_training.pdf, fig5_robustness.pdf

Fig. 3: nominal evaluation, (a) CDF of the per-episode 3-D error, (b) final X-Y error.
Fig. 4: training curves, mean ± SD over seeds, read from the TensorBoard logs of each run.
Fig. 5: stable success under the perturbations, mean and min-max over seeds.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.experiment import ACTION_SCALES_M, NOISE_STD, RADII_M  # noqa: E402
from evaluation.training_plots import load_run_scalars  # noqa: E402

METHODS = (("ppo", "PPO", "#c62828", "o"), ("divc-ppo", "DivC-PPO", "#2166ac", "s"))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def cells(results, run, condition):
    return sorted((results / "eval").glob(f"{run}_seed*/{condition}"))


def finish(fig, axes, out, ncol, top, extra=()):
    for ax, tag in zip(axes, "abc"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({tag})", fontsize=8, pad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    for handle, label in extra:
        handles.append(handle)
        labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, handlelength=1.6,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0, 1, top))
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print("wrote", out)


def fig3(results):
    fig, (ax_cdf, ax_xy) = plt.subplots(1, 2, figsize=(3.45, 1.95))
    for run, name, color, _ in METHODS:
        recs = [np.load(d / "inference_records.npz") for d in cells(results, run, "nominal")]
        for r in recs:
            x = np.sort(r["norm_mm"])
            ax_cdf.step(x, np.arange(1, len(x) + 1) / len(x), color=color, alpha=0.35, lw=0.5, where="post")
        pooled = np.sort(np.concatenate([r["norm_mm"] for r in recs]))
        ax_cdf.step(pooled, np.arange(1, len(pooled) + 1) / len(pooled), color=color, lw=1.2,
                    where="post", label=name)
        ex = np.concatenate([r["ex_mm"] for r in recs])
        ey = np.concatenate([r["ey_mm"] for r in recs])
        ax_xy.scatter(ex, ey, s=1.5, color=color, alpha=0.25, linewidths=0, rasterized=True)
        ax_xy.plot(ex.mean(), ey.mean(), marker="X", color=color, ms=6, mec="white", mew=0.8, zorder=5)
    ax_cdf.axvline(1.0, color="0.4", ls="--", lw=0.6)
    ax_cdf.set(xlim=(0, 4), ylim=(0, 1), xlabel="3-D error (mm)", ylabel="Fraction of episodes")
    ax_xy.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls="--", lw=0.6, ec="0.4"))
    ax_xy.axhline(0, color="0.8", lw=0.4, zorder=0)
    ax_xy.axvline(0, color="0.8", lw=0.4, zorder=0)
    ax_xy.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2],
              xlabel="$e_x$ (mm)")
    ax_xy.set_ylabel("$e_y$ (mm)", labelpad=1)
    mean_marker = Line2D([], [], ls="none", marker="X", color="0.3", ms=5, mec="white", mew=0.6)
    finish(fig, (ax_cdf, ax_xy), results / "fig3_eval_error.pdf", 3, 0.9, [(mean_marker, "Mean error")])


def fig4(results):
    manifest = results / "runs.json"
    if not manifest.is_file():
        print(f"skip Fig. 4: {manifest} not found")
        return
    runs = json.loads(manifest.read_text())
    panels = (("task/gt_error_norm_m", 1000.0, "3-D error (mm)", True),
              ("rollout/ep_rew_mean", 1.0, "Episode return", False))
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75))
    for run, name, color, _ in METHODS:
        scalars = [load_run_scalars(str((ROOT / path).parent)) for key, path in sorted(runs.items())
                   if key.rsplit("_seed", 1)[0] == run]
        for ax, (tag, scale, ylabel, logy) in zip(axes, panels):
            n = min(len(s[tag][0]) for s in scalars)  # a resumed run can log a few extra points
            steps = scalars[0][tag][0][:n]
            vals = np.stack([s[tag][1][:n] for s in scalars]) * scale
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros_like(mean)
            ax.plot(steps / 1e6, mean, color=color, lw=1.0, label=name)
            ax.fill_between(steps / 1e6, np.maximum(mean - sd, 1e-6 if logy else -np.inf), mean + sd,
                            color=color, alpha=0.2, lw=0)
            ax.set(xlim=(0, 1.02), xlabel="Environment steps (millions)", ylabel=ylabel)
            if logy:
                ax.set_yscale("log")
    finish(fig, axes, results / "fig4_training.pdf", 2, 0.9)


def fig5(results):
    groups = (("Noise (px)", [(float(s) * 640, f"noise_{s}") for s in NOISE_STD], [0, 0.5, 1, 1.5]),
              ("Radius (cm)", [(float(r) * 100, f"ood_r{r}") for r in RADII_M], [6, 10, 14]),
              ("Action scale (cm)", [(float(a) * 100, f"scale_{a}") for a in ACTION_SCALES_M], [2, 3, 4]))
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.45), sharey=True)
    for ax, (xlabel, points, ticks) in zip(axes, groups):
        xs = np.array([x for x, _ in points])
        for run, name, color, marker in METHODS:
            v = np.array([[100 * json.loads((d / "inference_metrics.json").read_text())
                           ["inference_deterministic"]["success_rate"] for d in cells(results, run, c)]
                          for _, c in points])
            mean = v.mean(axis=1)
            # Range over seeds, not ±SD: with n = 3 the SD can exceed the mean.
            ax.errorbar(xs, mean, yerr=np.vstack([mean - v.min(axis=1), v.max(axis=1) - mean]), color=color,
                        marker=marker, ms=2.5, lw=0.9, elinewidth=0.5, capsize=1.2, label=name)
        ax.set(xlabel=xlabel, ylim=(-5, 105), xticks=ticks)
    axes[0].set_ylabel("Stable success (%)")
    finish(fig, axes, results / "fig5_robustness.pdf", 2, 0.86)


if __name__ == "__main__":
    results = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results")
    fig3(results)
    fig4(results)
    fig5(results)
