"""Paper Tables III-V from the evaluation outputs of tools/reproduce.py (host-side, numpy only).

Reads <results>/eval/<run>_seed<seed>/<condition>/inference_metrics.json, where <run> is
ppo, divc-ppo, shared-ratio or scrambled. Every cell is the mean over seeds; Tables III and IV
add the sample standard deviation across seeds (ddof=1), as in the paper.

    python3 tools/tables.py [results]          # writes <results>/tables.md and prints it
"""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = {"ppo": "PPO", "divc-ppo": "DivC-PPO", "shared-ratio": "Shared-ratio control", "scrambled": "Scrambled mapping"}
FAMILIES = (("noise_", "Noise (px)", lambda v: f"{float(v) * 640:.3g}"),
            ("ood_r", "Radius (cm)", lambda v: f"{float(v) * 100:.0f}"),
            ("scale_", "Scale (cm)", lambda v: f"{float(v) * 100:.1f}"))


def load(results):
    """{(run, condition): {seed: metrics}} for complete validations."""
    cells = {}
    for path in sorted(Path(results).glob("eval/*_seed*/*/inference_metrics.json")):
        run, seed = path.parent.parent.name.rsplit("_seed", 1)
        m = json.loads(path.read_text())
        if m.get("validation_complete"):
            cells.setdefault((run, path.parent.name), {})[int(seed)] = m["inference_deterministic"]
    return cells


def per_seed(cell):
    """Per-seed (3-D error mm, stable success %, near-target failure %, time to stable success s)."""
    rows = []
    for seed in sorted(cell):
        d = cell[seed]
        rows.append((d["norm_mm"]["mean"], 100 * d["success_rate"], 100 * d["near_fail_rate"],
                     d["steps_to_stay20"]["mean"] * d["control_dt_s"]))
    return np.array(rows)


def pm(values, digits):
    sd = f" ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 else ""
    return f"{np.mean(values):.{digits}f}{sd}"


def table_iii(cells):
    lines = ["Table III. Nominal evaluation", "",
             "| Metric | PPO | DivC-PPO |", "|---|---:|---:|"]
    data = {run: per_seed(cells[(run, "nominal")]) for run in ("ppo", "divc-ppo") if (run, "nominal") in cells}
    for i, (name, digits) in enumerate((("3-D error (mm)", 3), ("Stable success (%)", 2),
                                        ("Near-target failure (%)", 2), ("Time to stable success (s)", 2))):
        lines.append(f"| {name} | " + " | ".join(pm(data[r][:, i], digits) if r in data else "–"
                                               for r in ("ppo", "divc-ppo")) + " |")
    return lines


def table_iv(cells):
    lines = ["Table IV. Mechanism controls (nominal)", "",
             "| Run | 3-D error (mm) | Stable success (%) |", "|---|---:|---:|"]
    for run in ("ppo", "divc-ppo", "shared-ratio", "scrambled"):
        if (run, "nominal") in cells:
            d = per_seed(cells[(run, "nominal")])
            lines.append(f"| {RUNS[run]} | {pm(d[:, 0], 3)} | {pm(d[:, 1], 2)} |")
    return lines


def table_v(cells):
    lines = ["Table V. Deployment perturbations (mean over seeds)", "",
             "| Condition | Level | PPO error (mm) | PPO success (%) | DivC-PPO error (mm) | DivC-PPO success (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    conditions = sorted({c for (_, c) in cells if c != "nominal"})
    for prefix, name, level in FAMILIES:
        for cond in sorted((c for c in conditions if c.startswith(prefix)), key=lambda c: float(c[len(prefix):])):
            row = []
            for run in ("ppo", "divc-ppo"):
                d = per_seed(cells[(run, cond)]) if (run, cond) in cells else None
                row += ["–", "–"] if d is None else [f"{d[:, 0].mean():.3f}", f"{d[:, 1].mean():.1f}"]
            lines.append(f"| {name} | {level(cond[len(prefix):])} | " + " | ".join(row) + " |")
    return lines


def main(results="results"):
    cells = load(results)
    if not cells:
        raise SystemExit(f"No complete inference_metrics.json under {results}/eval")
    text = "\n".join(table_iii(cells) + [""] + table_iv(cells) + [""] + table_v(cells)) + "\n"
    (Path(results) / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
