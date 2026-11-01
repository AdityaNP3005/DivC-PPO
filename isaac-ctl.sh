#!/bin/bash
# Dedicated DivC-PPO container; shared IsaacLab/venv, isolated project mount/output.
set -e
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-divc-ppo-runtime}"
# Directory whose storage/ holds the IsaacLab checkout and the Python venv (see README).
ISAAC_HOST_ROOT="${ISAAC_HOST_ROOT:?Set ISAAC_HOST_ROOT to the directory that contains storage/IsaacLab and storage/venv}"
STORAGE_HOST="$ISAAC_HOST_ROOT/storage"
PROJECT_MOUNT="/divc_ppo"
IMAGE_NAME="nvcr.io/nvidia/isaac-sim:5.1.0"

# Match the original launcher on accounts without Docker-socket access.
DOCKER=(docker)
if ! command docker info >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo -v; then
        DOCKER=(sudo docker)
        "${DOCKER[@]}" info >/dev/null
    else
        echo "Docker access failed. In your terminal run: sudo $BASE_DIR/isaac-ctl.sh ${1:-status}" >&2
        exit 1
    fi
fi
docker() { command "${DOCKER[@]}" "$@"; }

ensure_running() {
    docker inspect "$CONTAINER_NAME" --format '{{.State.Running}}' 2>/dev/null | grep -qx true || {
        echo "Start the runtime first: ./isaac-ctl.sh start" >&2; return 1;
    }
    local mounted
    mounted=$(docker inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/divc_ppo"}}{{.Source}}{{end}}{{end}}')
    [ "$mounted" = "$BASE_DIR" ] || {
        echo "Container mount mismatch: $mounted (expected $BASE_DIR)" >&2; return 1;
    }
    mounted=$(docker inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/storage"}}{{.Source}}{{end}}{{end}}')
    [ "$mounted" = "$STORAGE_HOST" ] || {
        echo "Shared dependency mount mismatch: $mounted (expected $STORAGE_HOST)" >&2; return 1;
    }
    if ! docker exec "$CONTAINER_NAME" nvidia-smi -L >/dev/null; then
        echo "GPU access failed inside container '$CONTAINER_NAME'; runtime is not ready." >&2
        echo "Use a fresh runtime via ISAAC_CONTAINER_NAME; see README.md for the recovery command." >&2
        return 1
    fi
}

start_sim() {
    [ -d "$STORAGE_HOST/IsaacLab/source/isaaclab" ] && \
        [ -d "$STORAGE_HOST/venv/lib/python3.11/site-packages" ] || {
        echo "Missing shared IsaacLab/venv under $STORAGE_HOST" >&2; return 1;
    }
    if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        [ "$(docker inspect "$CONTAINER_NAME" --format '{{.HostConfig.NetworkMode}}')" = host ] || {
            echo "Old DivC-PPO container has bridge networking. Run ./isaac-ctl.sh restart to migrate it." >&2
            return 1
        }
        docker start "$CONTAINER_NAME" >/dev/null
        ensure_running
    else
        mkdir -p "$BASE_DIR/logs/sb3" "$BASE_DIR/play_outputs" "$BASE_DIR/logs/isaac" "$BASE_DIR/.cache/isaac"
        docker run -d --name "$CONTAINER_NAME" --entrypoint bash \
        --gpus all --ipc=host --user "0:${SUDO_GID:-$(id -g)}" \
        --network=host \
        -e OMNI_KIT_ALLOW_ROOT=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
        -v "$BASE_DIR:$PROJECT_MOUNT:rw" -v "$STORAGE_HOST:/storage:ro" \
        -v "$BASE_DIR/.cache/isaac:/root/.cache:rw" \
        "$IMAGE_NAME" -c 'exec sleep infinity' >/dev/null
        ensure_running
    fi
    echo "Container: $CONTAINER_NAME"
    echo "Project:   $BASE_DIR -> $PROJECT_MOUNT (read/write)"
    echo "IsaacLab:  $STORAGE_HOST -> /storage (read-only)"
    docker exec "$CONTAINER_NAME" bash -c '
        for p in /isaac-sim/python.sh /isaac-sim/runheadless.sh /storage/IsaacLab/source/isaaclab /storage/venv/lib/python3.11/site-packages /divc_ppo/train_ur10e.py; do
            test -e "$p" || { echo "Missing container path: $p" >&2; exit 1; }
        done'
    if [ "${1:-}" = --runtime-only ]; then
        echo "Runtime ready for training/play/inference."
        return
    fi
    start_streaming
}

start_streaming() {
    if docker exec "$CONTAINER_NAME" pgrep -f '[i]saac-sim/kit/kit.*streaming' >/dev/null; then
        echo "Isaac streaming is already running."
    else
        local server_ip gpu_used
        server_ip=$(hostname -I | awk '{print $1}')
        if ss -ltn | awk '{print $4}' | grep -Eq ':49100$'; then
            echo "Streaming port 49100 is occupied. Stop the owning service first." >&2
            return 1
        fi
        gpu_used=$(nvidia-smi -i 0 --query-compute-apps=used_memory --format=csv,noheader,nounits |
            awk '{s+=$1} END {print s+0}')
        [ "$gpu_used" -le "${ISAAC_GPU_BUSY_MIB:-8000}" ] || {
            echo "GPU 0 busy ($gpu_used MiB); streaming launch refused." >&2; return 75;
        }
        mkdir -p "$BASE_DIR/logs/isaac"
        docker exec -d -e "STREAM_SERVER_IP=$server_ip" "$CONTAINER_NAME" bash -c '
            umask 0002
            cd /isaac-sim
            exec ./runheadless.sh \
                --/app/livestream/publicEndpointAddress="$STREAM_SERVER_IP" \
                --/app/livestream/port=49100 \
                --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false \
                --/physics/cudaDevice=0 \
                > /divc_ppo/logs/isaac/streaming.log 2>&1'
        echo "Isaac streaming launched; initialization takes about 30–60 seconds."
    fi
    echo "Streaming server: $(hostname -I | awk '{print $1}') (signaling TCP 49100)"
    echo "Connect with an Isaac-compatible streaming client; this image has no browser page on port 8211."
    echo "Logs:   $BASE_DIR/logs/isaac/streaming.log"
}

stop_job() {
    [ -n "${JOB_TAG:-}" ] || return 0
    docker exec -e "KILL_TAG=$JOB_TAG" "$CONTAINER_NAME" bash -lc '
        for p in /proc/[0-9]*; do
            grep -qzFx "DIVC_JOB=$KILL_TAG" "$p/environ" 2>/dev/null || continue
            kill -9 "${p#/proc/}" 2>/dev/null || true
        done' 2>/dev/null || true
    [ -z "${CLIENT_PID:-}" ] || kill "$CLIENT_PID" 2>/dev/null || true
    JOB_TAG=""
}

exec_cmd() {
    ensure_running
    local gpu_idx gpu_used lock_fd rc
    local key
    local forward_keys=(
        PPO_HEAD ABLATION10_ARM ABLATION_PROTOCOL DECOMPOSED_ADVANTAGE
        SINGLE_DECOMPOSED_ADVANTAGE DECOMPOSED_AXIS_WEIGHTS AXIS_CREDIT_PROBE
        SHARED_RATIO AXIS_CREDIT_PERMUTATION SCALAR_CRITIC SHARED_CRITIC
        ASYMMETRIC_CRITIC PPO_REWARD_UNITS DIVC_REWARD_UNITS REWARD_ERROR_SOURCE
        SEED NUM_ENVS PPO_N_STEPS PPO_BATCH_SIZE MAX_STEPS RL_CONTROL_MODE
        POSITION_SCALE_M GT_SUCCESS_XY_TOL_M GT_SUCCESS_Z_TOL_M GT_FINE_EXP_XY_SCALE_M
        PPO_LOG_STD_INIT PPO_ENT_COEF YOLO_MODEL_PATH OBJECT_USD OBJECT_USD_SCALE
        OBJECT_CYL_HEIGHT_M CYLINDER_FLOOR_Z_M YOLO_ROI_ENABLED
        YOLO_TRAIN_BBOX_CENTER_JITTER_STD YOLO_OBS_NOISE_STD PLAY_OBJECT_RADIUS_M
        EVAL_NUM_ENVS EVAL_MIN_EPISODES EVAL_MAX_STEPS ABLATION_EVAL_CONDITION
    )
    local forward_env=()
    for key in "${forward_keys[@]}"; do
        if [[ -v "$key" ]]; then forward_env+=(-e "$key=${!key}"); fi
    done
    gpu_idx="${GPU:-$(python3 -c 'from config.training import TRAIN_DEFAULTS; print(TRAIN_DEFAULTS["gpu"])')}"
    gpu_used=$(nvidia-smi -i "$gpu_idx" --query-compute-apps=used_memory --format=csv,noheader,nounits |
        awk '{s+=$1} END {print s+0}')
    if [ "$gpu_used" -gt "${ISAAC_GPU_BUSY_MIB:-8000}" ]; then
        echo "GPU $gpu_idx busy ($gpu_used MiB); launch refused." >&2; return 75
    fi
    # One lock per GPU: separate containers still share the physical GPU.
    exec {lock_fd}>"${ISAAC_GPU_JOB_LOCK_FILE:-/tmp/isaac-gpu-${gpu_idx}.lock}"
    flock -n "$lock_fd" || { echo "GPU $gpu_idx already locked." >&2; return 75; }
    JOB_TAG="divc-$$-$(date +%s)"
    trap 'stop_job' EXIT
    trap 'stop_job; exit 130' INT
    trap 'stop_job; exit 143' TERM HUP
    docker exec "${forward_env[@]}" -e "DIVC_JOB=$JOB_TAG" -e "GPU=$gpu_idx" \
        -w "$PROJECT_MOUNT" "$CONTAINER_NAME" bash -c 'umask 0002; exec "$@"' bash "$@" &
    CLIENT_PID=$!
    rc=0
    wait "$CLIENT_PID" || rc=$?
    [ "$rc" -eq 0 ] || stop_job
    trap - EXIT INT TERM HUP
    JOB_TAG=""; CLIENT_PID=""
    exec {lock_fd}>&-
    return "$rc"
}

run_isaaclab() {
    local script="$1"; shift
    local host_script
    if [[ "$script" == /* ]]; then
        host_script=$(realpath "$script")
    else
        host_script=$(realpath "$BASE_DIR/$script")
    fi
    [[ "$host_script" == "$BASE_DIR/"* ]] && [ -f "$host_script" ] || {
        echo "Script must exist inside this project: $script" >&2; return 2;
    }
    script="$PROJECT_MOUNT/${host_script#"$BASE_DIR/"}"
    local extra=()
    if [[ "$script" == */play.py || "$script" == */validate_inference.py ]]; then
        extra=(--headless --enable_cameras)
    fi
    exec_cmd bash -c '
        export PYTHONPATH="/divc_ppo:/storage/IsaacLab/source/isaaclab:/storage/IsaacLab/source/isaaclab_assets:/storage/IsaacLab/source/isaaclab_rl:/storage/venv/lib/python3.11/site-packages:${PYTHONPATH:-}"
        export HYDRA_FULL_ERROR=1 ENABLE_CAMERAS=1
        exec /isaac-sim/python.sh /divc_ppo/tools/run_runtime.py "$@"' bash "$script" "${extra[@]}" "$@"
}

cd "$BASE_DIR"
command="${1:-status}"; [ $# -eq 0 ] || shift
case "$command" in
    start) start_sim "$@" ;;
    stop) docker stop "$CONTAINER_NAME" ;;
    restart)
        if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
            docker stop "$CONTAINER_NAME" >/dev/null
            docker rm "$CONTAINER_NAME" >/dev/null
        fi
        start_sim "$@" ;;
    status)
        docker inspect "$CONTAINER_NAME" --format '{{.Name}} {{.State.Status}}'
        if docker exec "$CONTAINER_NAME" pgrep -f '[i]saac-sim/kit/kit.*streaming' >/dev/null 2>&1; then
            echo "Isaac application: streaming process running"
        else
            echo "Isaac application: stopped (runtime container only)"
        fi ;;
    logs)
        if [ -f "$BASE_DIR/logs/isaac/streaming.log" ]; then
            tail -n 100 "$BASE_DIR/logs/isaac/streaming.log"
        else
            docker logs --tail 100 "$CONTAINER_NAME"
        fi ;;
    exec) exec_cmd "$@" ;;
    run_isaaclab) run_isaaclab "$@" ;;
    *) echo "Usage: $0 {start|stop|restart|status|logs|exec|run_isaaclab}" >&2; exit 2 ;;
esac
