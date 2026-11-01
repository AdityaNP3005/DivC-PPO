"""Pre-Kit training setup: resolve PPO/DivC-PPO runtime arguments and filter noisy Isaac output.

Nothing here imports Isaac; it all runs before AppLauncher."""

import argparse
import os
import threading
from pathlib import Path

from evaluation.agent_metadata import parse_agent_yaml

from config.policy import scalar_critic_of, checkpoint_actor_depth, validate_supported_architecture
from config.control import require_cartesian_mode, validate_cartesian_checkpoint
from config.reward_units import reward_units_of
from config import (
    ASYMMETRIC_CRITIC_DEFAULTS,
    CAMERA_DEFAULTS,
    PPO_DEFAULTS,
    CREDIT_DEFAULTS,
    RESUME_DEFAULTS,
    TRAIN_DEFAULTS,
    YOLO_DEFAULTS,
    env_bool,
    publish_yolo_env,
)
from config.env_flags import expose_gpu
from config.policy import publish_run_mode


def _parse_ppo_head(text: str) -> str:
    head = str(text).lower().replace("-", "_").replace(" ", "_")
    if head == "single":
        return head
    raise argparse.ArgumentTypeError("Only PPO_head=single is supported")


def resolve_runtime_args(args_cli) -> None:
    """Populate args_cli + os.environ from defaults and forwarded ablation env vars.

    Mutates args_cli IN PLACE; must run BEFORE AppLauncher (sets CUDA_VISIBLE_DEVICES
    before CUDA init). Reads args_cli.checkpoint (may be None); sets everything else.
    """
    args_cli.num_envs = int(TRAIN_DEFAULTS["num_envs"])
    args_cli.max_steps = int(TRAIN_DEFAULTS["max_steps"])
    args_cli.checkpoint_interval = TRAIN_DEFAULTS["checkpoint_interval"]
    args_cli.keep_checkpoints = int(TRAIN_DEFAULTS["keep_checkpoints"])
    args_cli.checkpoint = args_cli.checkpoint or TRAIN_DEFAULTS["checkpoint"]
    if args_cli.checkpoint and os.environ.get("ABLATION_PROTOCOL"):
        raise ValueError("Matched baseline presets require fresh training; resume through the standalone entrypoint.")
    # The SEED env var overrides TRAIN_DEFAULTS["seed"] (used by tools/reproduce.py).
    # Empty/unset/non-int -> TRAIN_DEFAULTS.
    _seed_env = (os.environ.get("SEED") or "").strip()
    try:
        args_cli.seed = int(_seed_env) if _seed_env != "" else int(TRAIN_DEFAULTS["seed"])
    except ValueError:
        args_cli.seed = int(TRAIN_DEFAULTS["seed"])
        print(f"[SEED] ignoring non-integer SEED env '{_seed_env}'; using TRAIN_DEFAULTS seed={args_cli.seed}")
    # Preset env vars (config/experiment.py) override TRAIN_DEFAULTS. Empty/unset -> TRAIN_DEFAULTS.
    args_cli.mode = require_cartesian_mode(os.environ.get("RL_CONTROL_MODE"))
    args_cli.ppo_head = _parse_ppo_head(os.environ.get("PPO_HEAD") or TRAIN_DEFAULTS.get("PPO_head", "single"))
    _decomposed_requested = bool(CREDIT_DEFAULTS.get("decomposed_advantage", True))
    args_cli.single_decomposed_advantage = env_bool("SINGLE_DECOMPOSED_ADVANTAGE", True)
    args_cli.decomposed_advantage = bool(
        _decomposed_requested
        and args_cli.single_decomposed_advantage
    )
    # Per-axis credit on ONE scalar value head = the default (env SCALAR_CRITIC=0 widens it
    # to the 3-dim per-axis critic, an option the paper does not use).
    args_cli.scalar_critic = bool(
        args_cli.decomposed_advantage and CREDIT_DEFAULTS.get("scalar_critic", True)
    )
    args_cli.divc_reward_units = CREDIT_DEFAULTS["reward_units"]
    args_cli.scalar_reward_units = TRAIN_DEFAULTS["scalar_reward_units"]
    args_cli.asymmetric_critic = env_bool("ASYMMETRIC_CRITIC", bool(TRAIN_DEFAULTS.get("asymmetric_critic", False)))
    validate_supported_architecture({"ppo_head": args_cli.ppo_head, "z_control_mode": os.environ.get("Z_CONTROL_MODE") or "policy", "gate_mode": os.environ.get("GATE_MODE") or "rule"})
    args_cli.actor_depth_enabled = True
    if args_cli.checkpoint is not None:
        _agent_path = Path(args_cli.checkpoint).resolve().parent / "params" / "agent.yaml"
        if _agent_path.exists():
            try:
                with open(_agent_path, "r", encoding="utf-8") as _f:
                    _agent_text = _f.read()
                _agent_cfg_early = parse_agent_yaml(_agent_text) or {}
                if isinstance(_agent_cfg_early, dict):
                    validate_supported_architecture(_agent_cfg_early)
                    args_cli.actor_depth_enabled = checkpoint_actor_depth(_agent_cfg_early)
                    args_cli.asymmetric_critic = bool(_agent_cfg_early.get("asymmetric_critic", False))
                    args_cli.decomposed_advantage = bool(
                        _agent_cfg_early.get("decomposed_advantage", False)
                    )
                    args_cli.single_decomposed_advantage = bool(
                        args_cli.ppo_head == "single" and args_cli.decomposed_advantage
                    )
                    # Value-head WIDTH is baked into the checkpoint (1 vs 3), so resume
                    # must follow it, not the current env. Default False (NOT the config
                    # default): a checkpoint predating this key was trained 3-dim.
                    args_cli.scalar_critic = scalar_critic_of(_agent_cfg_early)
                    credit_cfg = (_agent_cfg_early.get("policy_kwargs", {}) or {}).get("decomposed_config", {})
                    args_cli.divc_reward_units = reward_units_of(credit_cfg)
                    args_cli.scalar_reward_units = _agent_cfg_early.get("scalar_reward_units", "per_step")
            except ValueError:
                raise
            except Exception as _exc:
                print(f"[ASYM] Failed to read checkpoint asymmetric flag: {_agent_path}: {_exc}")
    validate_cartesian_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint:
        from evaluation.checkpoint_assets import resolve_vecnormalize_path
        resolve_vecnormalize_path(args_cli.checkpoint)
    args_cli.privileged_obs_dim = int(ASYMMETRIC_CRITIC_DEFAULTS.get("privileged_obs_dim", 6))
    args_cli.gpu = int(TRAIN_DEFAULTS["gpu"])
    args_cli.camera_timeout_sec = float(TRAIN_DEFAULTS["camera_timeout_sec"])

    args_cli.ppo_n_steps = PPO_DEFAULTS["n_steps"]
    args_cli.ppo_batch_size = PPO_DEFAULTS["batch_size"]
    args_cli.ppo_n_epochs = PPO_DEFAULTS["n_epochs"]

    args_cli.resume_learning_rate = RESUME_DEFAULTS["learning_rate"]
    args_cli.resume_clip_range = RESUME_DEFAULTS["clip_range"]
    args_cli.resume_ent_coef = RESUME_DEFAULTS["ent_coef"]
    args_cli.resume_vf_coef = RESUME_DEFAULTS["vf_coef"]
    args_cli.resume_n_epochs = RESUME_DEFAULTS["n_epochs"]
    args_cli.resume_batch_size = RESUME_DEFAULTS["batch_size"]
    args_cli.resume_target_kl = RESUME_DEFAULTS["target_kl"]
    args_cli.resume_log_std_init = RESUME_DEFAULTS["log_std_init"]

    args_cli.camera_output_dir = CAMERA_DEFAULTS["output_dir"]
    args_cli.camera_resolution = CAMERA_DEFAULTS["resolution"]
    args_cli.camera_update_period = float(CAMERA_DEFAULTS["update_period"])

    # Honor an explicit YOLO_MODEL_PATH env override (a detector for another object). Without this
    # the hardcoded default silently clobbers the per-object weight → the object trains with the
    # wrong (generic) detector. No silent fallback: an explicit path that is missing must fail later.
    from evaluation.checkpoint_assets import require_detector
    args_cli.yolo_model_path = require_detector(os.environ.get("YOLO_MODEL_PATH") or YOLO_DEFAULTS["model_path"])
    args_cli.yolo_conf = float(YOLO_DEFAULTS["conf"])
    args_cli.yolo_iou = float(YOLO_DEFAULTS["iou"])
    args_cli.yolo_device = YOLO_DEFAULTS["device"]
    args_cli.yolo_max_det = int(YOLO_DEFAULTS["max_det"])


    print("[SETTINGS] Runtime defaults loaded from config/")

    # Auto-scale checkpoint_interval when None (10% of max_steps, min 50K)
    if args_cli.checkpoint_interval is None:
        args_cli.checkpoint_interval = max(int(args_cli.max_steps * 0.1), 50_000)
        print(f"[INFO] checkpoint_interval auto-scaled: {args_cli.checkpoint_interval:,} steps (10% of max_steps)")
    else:
        # Warn when the interval exceeds max_steps
        if args_cli.checkpoint_interval > args_cli.max_steps:
            print(f"[WARNING] checkpoint_interval ({args_cli.checkpoint_interval:,}) > max_steps ({args_cli.max_steps:,}) -> no checkpoint will be saved!")

    # Set the GPU before Isaac Sim starts (CUDA_VISIBLE_DEVICES must precede CUDA init)
    expose_gpu(args_cli, "cuda:0")
    print(f"[GPU] Using GPU index {args_cli.gpu} (CUDA_VISIBLE_DEVICES={args_cli.gpu})")
    # Same env-config flags as the validator (rl_env.env.runtime reads them on import).
    publish_run_mode(args_cli)
    print(
        f"[MODE] control_mode={args_cli.mode} | model=PPO | PPO_head={args_cli.ppo_head} | "
        f"asym_critic={args_cli.asymmetric_critic} | "
        f"decomposed_advantage={args_cli.decomposed_advantage} | "
        f"scalar_critic={args_cli.scalar_critic}"
    )

    # Same detector settings as the validator.
    publish_yolo_env(args_cli)
    print(
        "[YOLO] runtime config -> "
        f"model={os.environ['YOLO_MODEL_PATH']}, "
        f"device={os.environ['YOLO_DEVICE']}, "
        f"conf={os.environ['YOLO_CONF']}, iou={os.environ['YOLO_IOU']}, "
        f"max_det={os.environ['YOLO_MAX_DET']}"
    )


def _install_stream_filter(source_fd: int, target_fd: int, ignored_substrings: list[str]) -> None:
    """Filter selected lines from a stream without hiding real errors."""
    try:
        read_fd, write_fd = os.pipe()
        original_fd = os.dup(target_fd)
        os.dup2(write_fd, target_fd)
        os.close(write_fd)

        def _pump():
            with os.fdopen(read_fd, "r", errors="replace") as stream:
                for line in stream:
                    if any(token in line for token in ignored_substrings):
                        continue
                    os.write(original_fd, line.encode(errors="replace"))

        threading.Thread(target=_pump, daemon=True).start()
    except Exception:
        # If the filter fails, training continues.
        pass
