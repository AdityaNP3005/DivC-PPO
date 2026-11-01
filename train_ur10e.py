"""
Train the UR10e positioning policy with Stable-Baselines3 PPO.

USD Scene : assets/usd/stage_1.usd
Env ID    : Isaac-UR10e-DivC-Reach-v0
Model dir : Isaac-UR10e-DivC-Reach-v0

Usage:
    ./isaac-ctl.sh run_isaaclab train_ur10e.py --headless
    ./isaac-ctl.sh run_isaaclab train_ur10e.py --headless --checkpoint /divc_ppo/logs/sb3/Isaac-UR10e-DivC-Reach-v0/<timestamp>/model.zip

Training defaults are in config/training.py.

Output (otomatis):
    /divc_ppo/logs/sb3/Isaac-UR10e-DivC-Reach-v0/<timestamp>/
        model.zip                   ← model final
        model_<N>_steps.zip         ← checkpoint (only the last 5 are kept)
        PPO_1/                      ← TensorBoard logs
        params/env.yaml             ← snapshot config environment
        params/agent.yaml           ← snapshot config PPO
"""

# ─── 1. Parse argumen SEBELUM import Isaac Sim ────────────────────────────────
import argparse
import sys
import os
from pathlib import Path

# Suppress Isaac Sim verbose warnings (interpolation, carb, etc)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMNI_LOG_LEVEL", "warning")

if os.environ.get("NOISE_EVAL_OUT", "").strip():
    raise SystemExit(
        "NOISE_EVAL_OUT was removed: training does not run the noise evaluation. "
        "Use YOLO_OBS_NOISE_STD=<sigma> ./run.sh validate --checkpoint <model.zip> "
        "--out <fresh-directory>; read inference_metrics.json instead of legacy noise JSONL."
    )

from evaluation.checkpoint_assets import bootstrap_training_assets
bootstrap_training_assets()

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Import rl_env from this project, not from shared storage
for _m in list(sys.modules):
    if _m == "rl_env" or _m.startswith("rl_env."):
        del sys.modules[_m]

import config  # noqa: F401 — preserve eager asset/default initialization before runtime args


from training.runtime_args import _install_stream_filter, resolve_runtime_args


parser = argparse.ArgumentParser(description="Train the UR10e positioning policy with SB3 PPO.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint to resume from once (overrides config/training.py).",
)
# ── Inject enable_cameras before parse_args() ───────────────────────────────
os.environ.setdefault("ENABLE_CAMERAS", "1")
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Runtime defaults live in config/training.py. CLI is intentionally thin:
# keep AppLauncher flags plus --checkpoint for one-off resume.
resolve_runtime_args(args_cli)

# Bbox-center augmentation is training-only; play/validator never set this flag
# (the noise evaluation uses YOLO_OBS_NOISE_STD through the validator).
os.environ["YOLO_TRAIN_BBOX_CENTER_JITTER_ACTIVE"] = "1"

# Launch Isaac Sim (must precede any omni/isaacsim import)
# Install warning filters only after pre-launch validation so rejection errors
# reach stderr synchronously instead of being lost when a filter thread exits.
_install_stream_filter(1, 1, ["No adjacent samples found for interpolation"])
_install_stream_filter(2, 2, ["No adjacent samples found for interpolation"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── 2. Imports after Isaac Sim starts ────────────────────────────────────────
from datetime import datetime

from stable_baselines3.common.callbacks import CallbackList

from isaaclab.utils.io import dump_yaml

from rl_env.env.episode_env import Sb3VecEnvWrapper

# ─── 3. Register custom task ─────────────────────────────────────────────────
from rl_env.project_config import (
    LOG_ROOT,
    MODEL_DIR_NAME,
    TRAIN_ENV_ID,
)

import rl_env  # noqa: F401  — registrasi gym env
from rl_env.env import CUSTOM_SCENE_USD, UR10eEnvCfg
from rl_env.env.factory import make_isaac_env

from training.camera_probe import _configure_env_camera_sensor, setup_training_camera
from training.model_builder import build_training_config
from training.run_artifacts import run_dir_name
from training.resume import _space_summary, _wrapper_chain, setup_model
from training.callbacks import (
    AxisCreditProbeCallback,
    RewardTermLoggerCallback,
    TrainingMonitorCallback,
)


# ─── 5. Setup logging path ───────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
log_dir = os.path.join(LOG_ROOT, MODEL_DIR_NAME, run_dir_name(args_cli, timestamp))
os.makedirs(log_dir, exist_ok=False)
print(f"[INFO] Log dir: {log_dir}")


def main():
    # ─── 6. Environment config ───────────────────────────────────────────────
    env_cfg = UR10eEnvCfg()
    _configure_env_camera_sensor(env_cfg, args_cli)

    # ─── 7. IsaacLab environment (shared builder: cfg → USD patch → gym.make) ──
    env = make_isaac_env(env_cfg, TRAIN_ENV_ID, num_envs=args_cli.num_envs, seed=args_cli.seed)
    print(f"[VERIFY] Base env wrapper chain: {' -> '.join(_wrapper_chain(env))}")
    print(f"[VERIFY] Base observation_space: {_space_summary(getattr(env, 'observation_space', None))}")
    print(f"[VERIFY] Base action_space: {_space_summary(getattr(env, 'action_space', None))}")

    # ─── 7b. The camera must be ready before training ─────────────────────────
    try:
        setup_training_camera(log_dir, env, args_cli, simulation_app)
    except Exception as e:
        print(f"[CAMERA] ✖ {e}")
        try:
            env.close()
        finally:
            simulation_app.close()
        raise SystemExit(1)

    # Wrap into an SB3-compatible vectorized env
    env = Sb3VecEnvWrapper(env)
    print(f"[VERIFY] Post-SB3 wrapper chain: {' -> '.join(_wrapper_chain(env))}")
    print(f"[VERIFY] SB3 observation_space: {_space_summary(getattr(env, 'observation_space', None))}")
    print(f"[VERIFY] SB3 action_space: {_space_summary(getattr(env, 'action_space', None))}")

    # ─── 8. Save a config snapshot ───────────────────────────────────────────
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)

    # 
    # Training reward uses GT xyz, so no training-time visual auto-tolerance is saved.

    # Save the command line used
    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # ─── 9-10. Agent config (optimizer/sampling; rewards live in rewards_cfg) → fresh or resumed model ──
    model, env = setup_model(
        env, args_cli=args_cli, params_dir=params_dir, log_dir=log_dir,
        **build_training_config(env, env_cfg, args_cli, params_dir),
    )

    # ─── 11. Callbacks ───────────────────────────────────────────────────────
    from training.run_artifacts import SmartCheckpointCallback

    # PPO: save_freq is counted in rollout steps (checkpoint_interval / num_envs)
    _save_freq = max(args_cli.checkpoint_interval // args_cli.num_envs, 1)
    checkpoint_cb = SmartCheckpointCallback(
        keep_last_n=args_cli.keep_checkpoints,
        save_freq=_save_freq,
        save_path=log_dir,
        name_prefix="model",
        save_replay_buffer=False,  # not used by PPO
        save_vecnormalize=True,
    )
    # Preserve the same monitor target for fresh and resumed runs.
    start_steps = getattr(model, "num_timesteps", 0)
    monitor_total = int(args_cli.max_steps)

    monitor_cb = TrainingMonitorCallback(
        model_type="PPO",
        print_interval_steps=1_024,
        total_steps=monitor_total,
        verbose=1,
    )
    monitor_cb._start_steps = int(start_steps)
    # ─── 12. Training ────────────────────────────────────────────────────────
    print(f"\n[INFO] Training: {args_cli.max_steps:,} steps | {args_cli.num_envs} envs | model={'PPO'}")
    print(f"[INFO] Env task: {TRAIN_ENV_ID}")
    print(f"[INFO] Model dir: {MODEL_DIR_NAME}")
    print(f"[INFO] USD Scene: {CUSTOM_SCENE_USD}")
    print(f"[INFO] Log dir   : {log_dir}")
    print(f"[INFO] TensorBoard: tensorboard --logdir {log_dir}\n")

    reward_log_cb = RewardTermLoggerCallback(log_interval_steps=20_000)
    callback_list = [checkpoint_cb, monitor_cb, reward_log_cb]
    # Diagnostic only. OFF by default so no existing run changes behaviour; the probe
    # adds a per-step reward decomposition, which is cheap but not free.
    from config.env_flags import env_bool
    if env_bool("AXIS_CREDIT_PROBE", False):
        print("[INFO] AXIS_CREDIT_PROBE=1 → logging corr(advantage, per-axis reward)")
        callback_list.append(AxisCreditProbeCallback())

    try:
        if args_cli.checkpoint is None:
            print("[INFO] No checkpoint: training starts from step 0")
        else:
            print("[INFO] Resuming from the checkpoint step (reset_num_timesteps=False)")
        model.learn(
            total_timesteps=args_cli.max_steps,
            callback=CallbackList(callback_list),
            reset_num_timesteps=args_cli.checkpoint is None,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training stopped by the user.")

    # ─── 13. Save the final model ──────────────────────────────────────────────
    final_path = os.path.join(log_dir, "model.zip")
    model.save(final_path)
    env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    from training.run_artifacts import write_training_result
    write_training_result(log_dir, model, args_cli)
    print(f"[INFO] Model tersimpan: {final_path}")

    # ─── 14. Tutup environment ────────────────────────────────────────────────
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        from evaluation.runtime_cleanup import close_failed_app
        close_failed_app(error)
        raise
