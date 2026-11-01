"""Train and evaluate every run behind the paper tables.

Runs: PPO and DivC-PPO, plus the shared-ratio and scrambled controls of DivC-PPO, each for
seeds 42, 44 and 46 (12 trainings of 1,015,808 environment steps). All runs get the nominal
evaluation (1,024 episodes); PPO and DivC-PPO also get the 16 perturbation conditions
(>=100 episodes). Finished steps are skipped, so the script can be stopped and restarted.
Each step is one Isaac Lab process started with `$ISAACLAB_SH -p` (default: isaaclab.sh on PATH).

    python3 tools/reproduce.py [--dry-run] [--only train|eval] [--seeds 42 44 46] [--results results]
Then: python3 tools/tables.py results && python3 tools/figures.py results
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.experiment import CONDITIONS, SEEDS, condition_env, method_env  # noqa: E402
from rl_env.project_config import MODEL_DIR_NAME  # noqa: E402

RUNS = {"ppo": ("ppo", None), "divc-ppo": ("divc-ppo", None),
        "shared-ratio": ("divc-ppo", "shared-ratio"), "scrambled": ("divc-ppo", "scrambled")}
PERTURBED = ("ppo", "divc-ppo")  # Table V covers the two methods only
ISAACLAB = os.environ.get("ISAACLAB_SH", "isaaclab.sh")


def isaaclab(script, args, env, dry):
    command = [ISAACLAB, "-p", script, *args]
    print("$ " + " ".join(f"{k}={v}" for k, v in env.items()) + " " + shlex.join(command), flush=True)
    if not dry:
        subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, **env})


def train(name, seed, runs, dry):
    key = f"{name}_seed{seed}"
    if key in runs:
        return
    logs = ROOT / "logs/sb3" / MODEL_DIR_NAME
    before = set(logs.glob("*/model.zip"))
    isaaclab("train_ur10e.py", ["--headless"], dict(method_env(*RUNS[name]), SEED=str(seed)), dry)
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
        isaaclab("evaluate.py", ["--checkpoint", model, "--seed", str(seed), "--out", str(out)],
                 condition_env(condition), dry)


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
                if key not in runs and not args.dry_run:
                    raise SystemExit(f"{key} has no trained model in {manifest}")
                evaluate(name, seed, runs.get(key, f"<{key} model.zip>"), results, args.dry_run)


if __name__ == "__main__":
    main()
