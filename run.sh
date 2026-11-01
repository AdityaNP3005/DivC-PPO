#!/bin/bash
# Host launcher; every Isaac job runs in the container started by isaac-ctl.sh.
#   ./run.sh train {ppo|divc-ppo} [--control shared-ratio|scrambled] [--seed N] [--dry-run] [--headless] [--skip_post_video]
#   ./run.sh play [<model.zip>]
#   ./run.sh validate <model.zip> [--condition NAME] [--seed N] [--out DIR]
#   ./run.sh reproduce [--dry-run] [--only train|eval]     # all runs behind Tables III-V
# Methods, controls and evaluation conditions are defined in config/experiment.py.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ISAAC_CTL="${DIVC_ISAAC_CTL:-$SCRIPT_DIR/isaac-ctl.sh}"  # override only for launcher tests
MOUNT=/divc_ppo

usage() {
    sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

start_runtime() { "$ISAAC_CTL" start --runtime-only; }

# Export a KEY=VALUE bundle printed by config.experiment.
load_env() {
    local bundle
    bundle=$(python3 -m config.experiment "$@")
    set -a; eval "$bundle"; set +a
    printf '%s\n' "$bundle"
}

container_path() {
    local path
    path=$(realpath -m "$1")
    if [[ "$path" == "$SCRIPT_DIR/"* ]]; then echo "$MOUNT/${path#"$SCRIPT_DIR/"}"; else echo "$1"; fi
}

# train -> play/video -> deterministic validation in one GPU-locked container job.
train() {
    local method="${1:-}" control_args=() args=() dry=0
    [ -n "$method" ] || usage
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --control) control_args=(--control "$2"); shift 2 ;;
            --seed) export SEED="$2"; shift 2 ;;
            --dry-run) dry=1; shift ;;
            *) args+=("$1"); shift ;;
        esac
    done
    load_env method "$method" "${control_args[@]}" > /dev/null
    if [ "$dry" = 1 ]; then
        python3 -m config.experiment method "$method" "${control_args[@]}"
        echo "SEED=${SEED:-42}"
        exit 0
    fi
    start_runtime
    echo "[HOST] train $method ${control_args[*]} seed=${SEED:-42} -> play/video -> deterministic validation"
    exec "$ISAAC_CTL" exec bash "$MOUNT/train_and_play.sh" "${args[@]}"
}

# Play videos into ./play_outputs/, then validation of the same model.
# Without a path the latest final model is used for both stages.
play() {
    local model_path="${1:-}"
    start_runtime
    if [ -z "$model_path" ]; then
        model_path=$(python3 - <<'PY'
from pathlib import Path
from rl_env.project_config import log_base
paths = sorted(Path(log_base()).glob("*/model.zip"))
if not paths:
    raise SystemExit("No final model.zip found; specify a checkpoint.")
print(paths[-1])
PY
        )
    fi
    model_path=$(container_path "$model_path")
    "$ISAAC_CTL" run_isaaclab play.py --checkpoint "$model_path"
    echo ""
    echo "[INFO] Deterministic validation of the same model..."
    "$ISAAC_CTL" run_isaaclab validate_inference.py --checkpoint "$model_path"
}

# Deterministic validation under one evaluation condition (default: nominal).
validate() {
    local model="${1:-}" condition=nominal args=()
    [ -n "$model" ] || usage
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --condition) condition="$2"; shift 2 ;;
            --out) args+=(--out "$(container_path "$2")"); shift 2 ;;
            *) args+=("$1"); shift ;;
        esac
    done
    load_env condition "$condition"
    start_runtime
    exec "$ISAAC_CTL" run_isaaclab validate_inference.py --checkpoint "$(container_path "$model")" "${args[@]}"
}

reproduce() { exec python3 tools/reproduce.py "$@"; }

command="${1:-}"
[ $# -eq 0 ] || shift
case "$command" in
    train|play|validate|reproduce) "$command" "$@" ;;
    *) usage ;;
esac
