#!/bin/bash
# Container-side chain: train → play/video → deterministic validation (one GPU lock).
# Launched by ./run.sh train through isaac-ctl.sh exec.
set -e

SCRIPT_DIR="/divc_ppo"
ISAACLAB_DIR="/storage/IsaacLab"
PYTHON="/isaac-sim/python.sh"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:/storage/venv/lib/python3.11/site-packages"
export HYDRA_FULL_ERROR=1

# Keep query failure visible to set -e; mapfile/process substitution alone masks it.
LAUNCH_SETTINGS="$($PYTHON -c 'from rl_env.project_config import TRAIN_ENV_ID, MODEL_DIR_NAME, log_base; from config import PLAY_DEFAULTS; print(TRAIN_ENV_ID, MODEL_DIR_NAME, log_base(), PLAY_DEFAULTS["num_episodes"], PLAY_DEFAULTS["video_length"], sep="\n")')"
mapfile -t _launch_values <<< "$LAUNCH_SETTINGS"
TRAIN_ENV_ID="${_launch_values[0]}"
MODEL_DIR_NAME="${_launch_values[1]}"
LOG_BASE="${_launch_values[2]}"
PLAY_NUM_EPISODES="${_launch_values[3]}"
PLAY_VIDEO_LENGTH="${_launch_values[4]}"

SKIP_POST_VIDEO=0
PLAY_CHECKPOINT=""
TRAIN_ARGS=()

require_value() {
    local opt="$1"
    local val="${2-}"
    if [ -z "$val" ] || [[ "$val" == --* ]]; then
        echo "[ERROR] Missing value for $opt"
        exit 2
    fi
}

settings_only_error() {
    local opt="$1"
    echo "[ERROR] $opt is set in /divc_ppo/config/training.py, not on the command line."
    exit 2
}

while [ $# -gt 0 ]; do
    arg="$1"
    case "$arg" in
        --skip_post_video)
            SKIP_POST_VIDEO=1
            shift
            ;;
        --checkpoint=*)
            TRAIN_ARGS+=("$arg")
            shift
            ;;
        --checkpoint)
            require_value "$arg" "${2-}"
            TRAIN_ARGS+=("$arg" "$2")
            shift 2
            ;;
        --play_checkpoint=*)
            PLAY_CHECKPOINT="${arg#*=}"
            shift
            ;;
        --play_checkpoint)
            require_value "$arg" "${2-}"
            PLAY_CHECKPOINT="$2"
            shift 2
            ;;
        --headless|--enable_cameras)
            TRAIN_ARGS+=("$arg")
            shift
            ;;
        --*)
            settings_only_error "$arg"
            ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "        Runtime settings are in /divc_ppo/config/training.py."
            exit 2
            ;;
    esac
done

echo "======================================================="
echo "  TRAIN & PLAY  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Env task : $TRAIN_ENV_ID"
echo "  Model dir: $MODEL_DIR_NAME"
echo "======================================================="

echo ""
echo "[1/3] Training..."
echo "  Train args: ${TRAIN_ARGS[*]:-(from config/training.py)}"
echo ""

CHAIN_TMP=$(mktemp -d /tmp/divc-chain-XXXXXX)
export TRAIN_RUN_RESULT="$CHAIN_TMP/training_result.json"
trap 'rm -rf "$CHAIN_TMP"' EXIT
cd "$ISAACLAB_DIR"
set +e
$PYTHON "$SCRIPT_DIR/tools/run_runtime.py" "$SCRIPT_DIR/train_ur10e.py" "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo ""
echo "[1/3] Training finished (exit=$TRAIN_EXIT)."

if [ "$SKIP_POST_VIDEO" = "1" ]; then
    echo "[INFO] --skip_post_video set; play and validation skipped."
    exit 0
fi

if [ -n "$PLAY_CHECKPOINT" ]; then
    LATEST_MODEL="$PLAY_CHECKPOINT"
else
    LATEST_MODEL=$($PYTHON "$SCRIPT_DIR/tools/read_training_result.py" "$TRAIN_RUN_RESULT")
fi

if [ -z "$LATEST_MODEL" ]; then
    echo "[ERROR] Training produced no new model.zip in $LOG_BASE"
    echo "        Play cancelled so that an older checkpoint is not used."
    exit 1
fi

echo "[2/3] Model for play: $LATEST_MODEL"
echo "[2/3] Play (${PLAY_NUM_EPISODES} episodes, ${PLAY_VIDEO_LENGTH} steps each, from config/training.py)..."

export ENABLE_CAMERAS=1
cd "$ISAACLAB_DIR"
set +e
$PYTHON "$SCRIPT_DIR/tools/run_runtime.py" "$SCRIPT_DIR/play.py" \
    --checkpoint "$LATEST_MODEL" \
    --headless \
    --enable_cameras
PLAY_EXIT=$?
set -e

# [3/3] Auto-inference validation — deterministic parallel eval (min_episodes 1000 -> 1024 completed) on the SAME
# checkpoint, nested into the model's play folder (play_outputs/<tag>*/inference_validation/).
echo ""
echo "[3/3] Deterministic validation (1,024 parallel episodes)..."
export ENABLE_CAMERAS=1
cd "$ISAACLAB_DIR"
set +e
$PYTHON "$SCRIPT_DIR/tools/run_runtime.py" "$SCRIPT_DIR/validate_inference.py" \
    --checkpoint "$LATEST_MODEL"
VALIDATE_EXIT=$?
set -e
if [ $VALIDATE_EXIT -ne 0 ]; then
    echo "[ERROR] Validation failed (exit=$VALIDATE_EXIT)."
fi

echo ""
echo "======================================================="
echo "  DONE  |  play=$PLAY_EXIT  infer=$VALIDATE_EXIT  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="
if [ "$PLAY_EXIT" -ne 0 ]; then exit "$PLAY_EXIT"; fi
exit "$VALIDATE_EXIT"
