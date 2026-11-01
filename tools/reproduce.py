"""Train and evaluate every run behind the paper tables (host-side driver for run.sh).

Runs: PPO and DivC-PPO, plus the shared-ratio and scrambled controls of DivC-PPO, each for
seeds 42, 44 and 46 (12 trainings of 1M steps). All runs get the nominal evaluation
(1,024 episodes); PPO and DivC-PPO also get the 16 perturbation conditions (>=100 episodes).
Finished steps are skipped, so the script can be stopped and restarted.

    python3 tools/reproduce.py [--dry-run] [--only train|eval] [--seeds 42 44 46] [--results results]
Then: python3 tools/tables.py results && python3 tools/figures.py results
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.experiment import CONDITIONS, SEEDS  # noqa: E402
from rl_env.project_config import MODEL_DIR_NAME  # noqa: E402

RUNS = {"ppo": ("ppo", None), "divc-ppo": ("divc-ppo", None),
        "shared-ratio": ("divc-ppo", "shared-ratio"), "scrambled": ("divc-ppo", "scrambled")}
PERTURBED = ("ppo", "divc-ppo")  # Table V covers the two methods only


def run_sh(args, env=None, dry=False):
    print("$ ./run.sh " + " ".join(args), flush=True)
    if not dry:
        subprocess.run([str(ROOT / "run.sh"), *args], cwd=ROOT, check=True, env={**os.environ, **(env or {})})


def train(name, seed, runs, dry):
    key = f"{name}_seed{seed}"
    if key in runs:
        return
    method, control = RUNS[name]
    logs = ROOT / "logs/sb3" / MODEL_DIR_NAME
    before = set(logs.glob("*/model.zip"))
    run_sh(["train", method, *(["--control", control] if control else []), "--seed", str(seed),
            "--headless", "--skip_post_video"], dry=dry)
    if dry:
        return
    new = sorted(set(logs.glob("*/model.zip")) - before)
    if len(new) != 1:
        raise SystemExit(f"{key}: expected one new model.zip, found {len(new)}")
    runs[key] = str(new[0].relative_to(ROOT))


def evaluate(name, seed, model, results, dry):
    for condition in CONDITIONS if name in PERTURBED else ("nominal",):
        out = Path(results) / "eval" / f"{name}_seed{seed}" / condition
        metrics = out / "inference_metrics.json"
        if metrics.is_file() and json.loads(metrics.read_text()).get("validation_complete"):
            continue
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} holds an incomplete evaluation; move it away and rerun")
        run_sh(["validate", model, "--condition", condition, "--seed", str(seed), "--out", str(out)], dry=dry)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=("train", "eval"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--runs", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results = ROOT / args.results
    manifest = results / "runs.json"
    runs = json.loads(manifest.read_text()) if manifest.is_file() else {}
    for seed in args.seeds:
        for name in args.runs:
            if args.only != "eval":
                train(name, seed, runs, args.dry_run)
                if not args.dry_run:
                    results.mkdir(parents=True, exist_ok=True)
                    manifest.write_text(json.dumps(runs, indent=2) + "\n")
            key = f"{name}_seed{seed}"
            if args.only != "train":
                model = runs.get(key, f"<{key} model.zip>")
                if key not in runs and not args.dry_run:
                    raise SystemExit(f"{key} has no trained model in {manifest}")
                evaluate(name, seed, model, results, args.dry_run)


if __name__ == "__main__":
    main()
