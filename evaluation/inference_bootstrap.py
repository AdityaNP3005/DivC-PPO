"""Config-safe pre-Kit replay stages shared by play and deterministic validation.

Checkpoint selection/object restoration belongs to config.training's earlier
bootstrap. These stages reuse its path; environment/model construction stays
with each entrypoint after AppLauncher starts.
"""
from dataclasses import dataclass
import os

from evaluation.checkpoint_assets import require_detector, resolve_vecnormalize_path
from evaluation.checkpoint_rebuild import _infer_ppo_head_from_agent_cfg, _load_checkpoint_agent_cfg_early


@dataclass(frozen=True)
class CheckpointContext:
    path: str | None
    agent: dict
    normalizer: str | None
    method: str | None


def apply_runtime_defaults(args, defaults):
    args.gpu = defaults["gpu"]
    args.camera_resolution = defaults["camera_resolution"]
    args.camera_update_period = float(defaults["camera_update_period"])
    args.yolo_model_path = defaults["yolo_model_path"]
    args.yolo_conf = float(defaults["yolo_conf"])
    args.yolo_iou = float(defaults["yolo_iou"])
    args.yolo_device = defaults["yolo_device"]
    args.yolo_max_det = int(defaults["yolo_max_det"])


def load_checkpoint_context(path, *, matched_validation=False, method=None):
    agent = _load_checkpoint_agent_cfg_early(path)
    matched = None
    if matched_validation:
        from evaluation.checkpoint_contract import require_checkpoint_method
        matched = require_checkpoint_method(path, agent, method)
    normalizer = resolve_vecnormalize_path(path) if path else None
    if normalizer:
        print(f"[CHECKPOINT] Matching VecNormalize: {normalizer}", flush=True)
    return CheckpointContext(path, agent, normalizer, matched)


def apply_checkpoint_model(args, checkpoint, training_defaults):
    from config.control import require_cartesian_mode, validate_cartesian_checkpoint
    args.mode = require_cartesian_mode(os.environ.get("RL_CONTROL_MODE"))
    validate_cartesian_checkpoint(checkpoint.path, checkpoint.agent)
    args.ppo_head = _infer_ppo_head_from_agent_cfg(checkpoint.agent)
    args.asymmetric_critic = bool(checkpoint.agent.get(
        "asymmetric_critic", bool(training_defaults.get("asymmetric_critic", False))))
    from config.policy import checkpoint_actor_depth
    args.actor_depth_enabled = checkpoint_actor_depth(checkpoint.agent)
    # Explicit process overrides win, including zero-shot object/detector replay.
    detector = os.environ.get("YOLO_MODEL_PATH") or checkpoint.agent.get("yolo_model_path")
    if detector:
        from rl_env.project_config import local_asset_path
        args.yolo_model_path = local_asset_path(str(detector))


def configure_inference_device(args):
    from config.env_flags import expose_gpu
    from config.policy import publish_run_mode
    # All in-process CUDA tensors use local cuda:0 once one physical GPU is exposed.
    if args.gpu is not None or str(getattr(args, "device", "cuda:0")) == "cuda":
        args.device = "cuda:0"
    run_device = str(getattr(args, "device", "cuda:0"))
    expose_gpu(args, run_device, flush=True)
    publish_run_mode(args)
    return run_device


def configure_inference_detector(args):
    args.yolo_model_path = require_detector(args.yolo_model_path)
    from config.perception import publish_yolo_env
    publish_yolo_env(args)


def load_inference_policy(base_env, checkpoint, normalizer, *, seed, device, simulation_app):
    """SB3 wrapper + frozen checkpoint VecNormalize + PPO for validation. Exits on mismatch.

    A missing normalizer or an observation-shape mismatch would silently feed the policy
    wrongly scaled/shaped inputs, so both abort the evaluation.
    """
    import sys

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from rl_env.env.episode_env import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(base_env)
    if not normalizer or not os.path.exists(normalizer):
        print(
            f"[ERROR] VecNormalize not found: {normalizer}. "
            "Evaluation cancelled so the checkpoint does not receive wrongly scaled observations.",
            flush=True,
        )
        env.close()
        simulation_app.close()
        sys.exit(1)
    env = VecNormalize.load(normalizer, env)
    env.training = False  # never update obs_rms during evaluation
    env.norm_reward = False
    print("[INFO] VecNormalize loaded (training=False).", flush=True)

    model = PPO.load(checkpoint, env=env, device=device)
    # PPO.load restores its training seed; reset the RNG to the eval seed before env.reset.
    base_env.unwrapped.seed(seed)
    env_shape = getattr(env.observation_space, "shape", None)
    model_shape = getattr(model.observation_space, "shape", None)
    if env_shape and model_shape and env_shape != model_shape:
        print(
            f"[ERROR] Observation space mismatch: env={env_shape} model={model_shape}. "
            "The env observation terms changed after this checkpoint was saved: retrain or revert them.",
            flush=True,
        )
        env.close()
        simulation_app.close()
        sys.exit(1)
    return env, model
