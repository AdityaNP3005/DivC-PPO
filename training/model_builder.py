"""Build single-actor PPO, asymmetric PPO, and DivC-PPO models."""

import math
import os

import numpy as np
from stable_baselines3 import PPO

from isaaclab.utils.io import dump_yaml

from rl_env.policy import (
    AsymmetricActorCriticPolicy,
    DivCPPO,
)
from config.policy import scalar_critic_of
from config.reward_units import reward_units_of, reward_error_source
from config import (
    PPO_DEFAULTS,
    CREDIT_DEFAULTS,
)


def build_agent_snapshot_cfg(
    cfg: dict,
    *,
    args_cli,
    actor_obs_dim,
    obs_dim,
    privileged_obs_dim,
) -> dict:
    def _plain(value):
        if isinstance(value, dict):
            return {str(k): _plain(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [_plain(v) for v in value]
        if isinstance(value, list):
            return [_plain(v) for v in value]
        return value

    snap = _plain(dict(cfg))
    snap["policy_kwargs"] = _plain(dict(cfg.get("policy_kwargs", {})))
    snap["ppo_head"] = args_cli.ppo_head
    snap["control_mode"] = "cartesian"
    snap["action_dim"] = 3
    snap["asymmetric_critic"] = bool(args_cli.asymmetric_critic)
    snap["decomposed_advantage"] = bool(args_cli.decomposed_advantage)
    # Value-head width under per-axis credit: False → 3 outputs, True → 1 shared output.
    snap["scalar_critic"] = bool(getattr(args_cli, "scalar_critic", False))
    snap["value_head_outputs"] = 3 if args_cli.decomposed_advantage and not snap["scalar_critic"] else 1
    snap["num_envs"] = int(args_cli.num_envs)
    snap["max_steps_requested"] = int(args_cli.max_steps)
    snap["reward_stats_units"] = "per_step"
    snap["reward_error_source"] = reward_error_source(args_cli.asymmetric_critic)
    snap["scalar_reward_units"] = args_cli.scalar_reward_units if not args_cli.decomposed_advantage else "environment"
    axis_cfg = snap["policy_kwargs"].get("decomposed_config", {})
    snap["decomposed_reward_units"] = reward_units_of(axis_cfg) if args_cli.decomposed_advantage else "environment"
    if os.environ.get("ABLATION_PROTOCOL"):
        snap["comparison_protocol"] = os.environ["ABLATION_PROTOCOL"]
    arm = (os.environ.get("ABLATION10_ARM") or "").strip()
    snap["ablation10_arm"] = int(arm) if arm else None
    snap["actor_depth_enabled"] = bool(args_cli.actor_depth_enabled)
    snap["actor_obs_dim"] = actor_obs_dim
    snap["critic_obs_dim"] = obs_dim
    snap["privileged_obs_dim"] = privileged_obs_dim
    snap["z_control_mode"] = "policy"  # explicit compatibility metadata
    # Persist the TRAINING OBJECT (any OBJECT_USD run) so a manual replay
    # (./run.sh play) spawns the SAME object at the SAME height/floor + loads the SAME per-object
    # YOLO — without needing the OBJECT_*/YOLO_MODEL_PATH env vars re-set. play.py restores these
    # from agent.yaml before config is imported. Empty object_usd → the default primitive cylinder.
    from config.task import (
        CYLINDER_FLOOR_Z_M,
        OBJECT_CYL_HEIGHT_M,
        OBJECT_USD,
        OBJECT_USD_SCALE,
    )

    snap["object_usd"] = str(OBJECT_USD)
    snap["object_usd_scale"] = float(OBJECT_USD_SCALE)
    snap["object_cyl_height_m"] = float(OBJECT_CYL_HEIGHT_M)
    snap["cylinder_floor_z_m"] = float(CYLINDER_FLOOR_Z_M)
    snap["yolo_model_path"] = os.environ.get("YOLO_MODEL_PATH", "")
    # Record filter provenance only. Replay always uses the current canonical mask;
    # old checkpoint rectangle geometry must never override it.
    from config import (
        YOLO_GRIPPER_EXCLUSION_MASK_VERSION,
        YOLO_GRIPPER_EXCLUSION_MODE,
        YOLO_ROI_ENABLED,
        YOLO_TRAIN_BBOX_CENTER_JITTER_STD,
    )

    snap["yolo_roi_enabled"] = bool(YOLO_ROI_ENABLED)
    snap["yolo_detection_filter_mode"] = str(YOLO_GRIPPER_EXCLUSION_MODE)
    snap["yolo_gripper_exclusion_mask_version"] = int(YOLO_GRIPPER_EXCLUSION_MASK_VERSION)
    # Training provenance only: replay intentionally stays clean unless the noise evaluation sets
    # YOLO_OBS_NOISE_STD explicitly.
    snap["yolo_train_bbox_center_jitter_std"] = float(YOLO_TRAIN_BBOX_CENTER_JITTER_STD)
    return snap


def apply_ppo_head_runtime(
    model_obj,
    env_obj,
    label: str,
    *,
    args_cli,
    actor_obs_dim,
    obs_dim,
    privileged_obs_dim,
) -> None:
    policy = getattr(model_obj, "policy", None)
    supports_asym = hasattr(policy, "set_actor_obs_dim")
    policy_axis_cfg = getattr(policy, "decomposed_config", None) or {}
    loaded_decomposed = bool(policy_axis_cfg.get("decomposed_advantage", False))
    requested_decomposed = bool(args_cli.decomposed_advantage)
    if loaded_decomposed != requested_decomposed:
        raise RuntimeError(
            "Cannot switch decomposed_advantage while resuming: the value head and rollout buffer differ. "
            "Replay the checkpoint as-is or start fresh training."
        )
    if loaded_decomposed and reward_units_of(policy_axis_cfg) != args_cli.divc_reward_units:
        raise RuntimeError("Cannot switch decomposed reward units while resuming.")
    if scalar_critic_of(policy_axis_cfg) != bool(getattr(args_cli, "scalar_critic", False)):
        raise RuntimeError(
            "Cannot switch scalar_critic while resuming: the value head width differs (1 vs 3 outputs). "
            "Replay the checkpoint as-is or start fresh training."
        )
    if args_cli.asymmetric_critic:
        if not supports_asym:
            raise RuntimeError(
                "asymmetric_critic=True but checkpoint/model policy does not support actor_obs_dim masking. "
                "Start fresh training or resume a checkpoint trained with asymmetric_critic=True."
            )
        policy.set_actor_obs_dim(actor_obs_dim)
        print(
            f"[ASYM] {label}: actor_obs_dim={actor_obs_dim}, "
            f"critic_obs_dim={obs_dim}, privileged_obs_dim={privileged_obs_dim}"
        )



def build_training_config(env, env_cfg, args_cli, params_dir):
    """Assemble the PPO/DivCPPO run config from the env + ablation flags.

    Auto-scales n_steps/batch_size, derives observation dimensions, picks the policy + algo class, prints the training
    banner and dumps params/agent.yaml. Returns the pieces main() needs downstream.
    Behaviour byte-identical to the old inline block.
    """
    _net_arch = PPO_DEFAULTS["net_arch"]

    # Auto-scale n_steps and batch_size with num_envs to keep rollouts manageable.
    # SB3 shows ep_rew_mean only after an episode has finished.
    # Episodes here last about 210 steps, so with n_steps=64 ep_rew_mean appears
    # after about ceil(210/64)=4 updates.
    if args_cli.num_envs <= 64:
        _n_steps = 256
        _batch_size = 1024
    elif args_cli.num_envs <= 128:
        _n_steps = 256
        _batch_size = 2048
    elif args_cli.num_envs <= 256:
        _n_steps = 256
        _batch_size = 4096
    else:  # 512+ envs
        _n_steps = 128
        _batch_size = 4096

    if args_cli.ppo_n_steps is not None:
        _n_steps = max(int(args_cli.ppo_n_steps), 8)
    if args_cli.ppo_batch_size is not None:
        _batch_size = max(int(args_cli.ppo_batch_size), 1)
    _ppo_n_epochs = max(int(args_cli.ppo_n_epochs), 1) if args_cli.ppo_n_epochs is not None else 3

    _rollout = _n_steps * args_cli.num_envs
    _batch_size = min(_batch_size, _rollout)
    print(f"[INFO] Auto-scaled n_steps={_n_steps}, batch_size={_batch_size} for num_envs={args_cli.num_envs}")
    print(f"[INFO] Rollout size = {_rollout:,} steps | n_epochs={_ppo_n_epochs}")
    if _rollout % _batch_size != 0:
        print(
            f"[WARNING] PPO rollout ({_rollout}) is not divisible by batch_size ({_batch_size}); "
            "SB3 will use a truncated last minibatch."
        )

    try:
        _episode_steps_est = int(math.ceil(float(env_cfg.episode_length_s) / (float(env_cfg.sim.dt) * float(env_cfg.decimation))))
    except Exception:
        _episode_steps_est = None
    if _episode_steps_est:
        _updates_until_ep_rew = max(1, int(math.ceil(_episode_steps_est / max(_n_steps, 1))))
        print(
            f"[INFO] Episode horizon ~= {_episode_steps_est} env steps; "
            f"ep_rew_mean appears after ~{_updates_until_ep_rew} PPO verbose/update(s)."
        )
        if _updates_until_ep_rew > 1:
            print(
                "[INFO] Untuk ep_rew_mean tiap verbose, set PPO_DEFAULTS['n_steps']=256 "
                "and batch_size 2048/4096 in config/training.py."
            )

    obs_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    obs_dim = int(np.prod(obs_shape)) if obs_shape is not None else 0
    privileged_obs_dim = int(args_cli.privileged_obs_dim) if args_cli.asymmetric_critic else 0
    actor_obs_dim = max(obs_dim - privileged_obs_dim, 1) if args_cli.asymmetric_critic else None
    if args_cli.asymmetric_critic:
        print(
            f"[ASYM] privileged critic active: actor_obs_dim={actor_obs_dim}, "
            f"critic_obs_dim={obs_dim}, privileged_obs_dim={privileged_obs_dim}"
        )

    policy_kwargs = {
        "net_arch": _net_arch,
        "log_std_init": PPO_DEFAULTS["log_std_init"],
    }
    decomposed_config = {
        "reward_units": args_cli.divc_reward_units,
        "decomposed_advantage": bool(args_cli.decomposed_advantage),
        "decomposed_axis_weights": CREDIT_DEFAULTS["decomposed_axis_weights"],
        "shared_ratio": bool(CREDIT_DEFAULTS.get("shared_ratio", False)),
        "axis_credit_permutation": str(CREDIT_DEFAULTS.get("axis_credit_permutation", "none")),
        "scalar_critic": bool(getattr(args_cli, "scalar_critic", False)),
    }
    policy_class = "MlpPolicy"
    if args_cli.asymmetric_critic:
        policy_class = AsymmetricActorCriticPolicy
        policy_kwargs["actor_obs_dim"] = actor_obs_dim
    if args_cli.decomposed_advantage:
        if args_cli.mode != "cartesian":
            raise RuntimeError("Decomposed advantage requires a 3D Cartesian action space.")
        policy_class = AsymmetricActorCriticPolicy
        policy_kwargs["decomposed_config"] = decomposed_config

    ppo_cfg = {
        "seed": args_cli.seed,
        "n_steps": _n_steps,
        "batch_size": _batch_size,
        "n_epochs": _ppo_n_epochs,
        "learning_rate": PPO_DEFAULTS["learning_rate"],
        "gamma": PPO_DEFAULTS["gamma"],
        "gae_lambda": PPO_DEFAULTS["gae_lambda"],
        "clip_range": PPO_DEFAULTS["clip_range"],
        "target_kl": PPO_DEFAULTS.get("target_kl", None),
        "ent_coef": PPO_DEFAULTS["ent_coef"],
        "vf_coef": PPO_DEFAULTS["vf_coef"],
        "max_grad_norm": PPO_DEFAULTS["max_grad_norm"],
        "policy_kwargs": policy_kwargs,
    }

    agent_cfg = ppo_cfg
    # DivCPPO owns the optional XYZ buffer/train path; stock single runs stay SB3 PPO.
    algo_cls = DivCPPO if args_cli.decomposed_advantage else PPO

    _rollout = ppo_cfg["n_steps"] * args_cli.num_envs
    print("[INFO] ┌─ Training Config (PPO) ─────────────────────")
    print(f"[INFO] │  num_envs   = {args_cli.num_envs}")
    print(f"[INFO] │  n_steps    = {ppo_cfg['n_steps']}")
    print(f"[INFO] │  rollout    = n_steps × num_envs = {_rollout:,}")
    print(f"[INFO] │  batch_size = {ppo_cfg['batch_size']:,}")
    print(f"[INFO] │  n_epochs   = {ppo_cfg['n_epochs']}  →  {_rollout // ppo_cfg['batch_size'] * ppo_cfg['n_epochs']} gradient steps/update")
    print(f"[INFO] │  PPO_head   = {args_cli.ppo_head}")
    print(
        f"[INFO] │  asym_crit = {args_cli.asymmetric_critic} "
        f"(actor_obs={actor_obs_dim}, critic_obs={obs_dim}, priv={privileged_obs_dim})"
    )
    print(
        "[INFO] │  advantage  = "
        f"decomposed_xyz={args_cli.decomposed_advantage} "
        f"scalar_critic={decomposed_config['scalar_critic']} "
        f"(value head {1 if decomposed_config['scalar_critic'] or not args_cli.decomposed_advantage else 3}-dim) "
        f"weights={decomposed_config['decomposed_axis_weights']} "
        f"shared_ratio={decomposed_config['shared_ratio']} "
        f"perm={decomposed_config['axis_credit_permutation']} algo={algo_cls.__name__}"
    )
    print("[INFO] └────────────────────────────────────────────────")

    dump_yaml(os.path.join(params_dir, "agent.yaml"), build_agent_snapshot_cfg(agent_cfg, args_cli=args_cli, actor_obs_dim=actor_obs_dim, obs_dim=obs_dim, privileged_obs_dim=privileged_obs_dim))

    return {
        "agent_cfg": agent_cfg,
        "algo_cls": algo_cls,
        "policy_class": policy_class,
        "actor_obs_dim": actor_obs_dim,
        "obs_dim": obs_dim,
        "privileged_obs_dim": privileged_obs_dim,
    }
