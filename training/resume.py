"""Model construction / checkpoint-resume for training (lifted out of main()).

Handles BOTH paths of the old `if args_cli.checkpoint is not None: ... else: ...`
block: fresh model creation and full checkpoint resume (VecNormalize load, PPO
hyperparameter override, asymmetric runtime wiring). Returns (model, env). Matching
normalization selection is shared with replay/validation; missing or corrupt selected
pairs fail without final-stat fallback. Also holds the resume diagnostics (config/
hyperparameter diff, VecNormalize state) and the [VERIFY] wrapper/space printers.
"""

import math
import os
from pathlib import Path

import numpy as np
import yaml
import torch
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.io import dump_yaml

from config import PPO_DEFAULTS

from training.model_builder import (
    apply_ppo_head_runtime,
    build_agent_snapshot_cfg,
)
from training.reward_normalization import PpoRewardVecNormalize


def _load_yaml_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        # IsaacLab env snapshots often contain Python-specific YAML tags such as
        # `!!python/tuple` and `!!python/object/apply:builtins.slice`. Those cannot
        # be parsed by `safe_load`, so we intentionally use a trusted loader here.
        try:
            if hasattr(yaml, "unsafe_load"):
                data = yaml.unsafe_load(f)
            else:
                unsafe_loader = getattr(yaml, "UnsafeLoader", None) or getattr(yaml, "Loader", None)
                if unsafe_loader is None:
                    data = yaml.full_load(f)
                else:
                    data = yaml.load(f, Loader=unsafe_loader)
        except Exception:
            f.seek(0)
            data = yaml.full_load(f)
    return data if isinstance(data, dict) else {}


def _wrapper_chain(obj, max_depth: int = 8) -> list[str]:
    chain: list[str] = []
    seen: set[int] = set()
    current = obj
    for _ in range(max_depth):
        if current is None:
            break
        cur_id = id(current)
        if cur_id in seen:
            chain.append(f"<cycle:{type(current).__name__}>")
            break
        seen.add(cur_id)
        chain.append(type(current).__name__)
        next_obj = None
        for attr in ("venv", "env", "unwrapped", "_env", "_wrapped_env"):
            if hasattr(current, attr):
                next_obj = getattr(current, attr)
                if next_obj is not None:
                    break
        current = next_obj
    return chain


def _space_summary(space) -> str:
    if space is None:
        return "None"
    shape = getattr(space, "shape", None)
    if hasattr(space, "low") and hasattr(space, "high"):
        try:
            low = np.asarray(space.low)
            high = np.asarray(space.high)
            return f"{type(space).__name__}(shape={shape}, low=[{float(np.min(low)):.4g},{float(np.max(low)):.4g}], high=[{float(np.min(high)):.4g},{float(np.max(high)):.4g}])"
        except Exception:
            return f"{type(space).__name__}(shape={shape})"
    return f"{type(space).__name__}(shape={shape})"


def _print_vecnormalize_state(vec_env, label: str = "VecNormalize") -> None:
    attrs = {
        "training": getattr(vec_env, "training", None),
        "norm_obs": getattr(vec_env, "norm_obs", None),
        "norm_reward": getattr(vec_env, "norm_reward", None),
        "clip_obs": getattr(vec_env, "clip_obs", None),
        "clip_reward": getattr(vec_env, "clip_reward", None),
        "gamma": getattr(vec_env, "gamma", None),
    }
    print(f"[VERIFY] {label}: " + ", ".join(f"{k}={v}" for k, v in attrs.items()))

    for name in ("obs_rms", "ret_rms"):
        rms = getattr(vec_env, name, None)
        if rms is None:
            print(f"[VERIFY] {label}.{name}=None")
            continue
        mean = getattr(rms, "mean", None)
        var = getattr(rms, "var", None)
        print(
            f"[VERIFY] {label}.{name}: mean_shape={None if mean is None else tuple(np.asarray(mean).shape)}, "
            f"std_shape={None if var is None else tuple(np.asarray(var).shape)}"
        )


def _stable_config_repr(value):
    """Return a deterministic, comparable representation for config snapshots."""
    if isinstance(value, dict):
        return {str(k): _stable_config_repr(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_config_repr(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _reward_config_signature(cfg: dict) -> dict:
    rewards = cfg.get("rewards", {}) if isinstance(cfg, dict) else {}
    if not isinstance(rewards, dict):
        return {}
    return {str(name): _stable_config_repr(term) for name, term in sorted(rewards.items(), key=lambda item: str(item[0]))}


def _reset_vecnormalize_reward_stats(vec_env, label: str = "VecNormalize") -> None:
    """Reset reward-return normalization only; keep obs_rms from checkpoint intact."""
    ret_rms = getattr(vec_env, "ret_rms", None)
    if ret_rms is not None:
        for attr, value in (("mean", 0.0), ("var", 1.0)):
            arr = getattr(ret_rms, attr, None)
            if arr is not None:
                try:
                    arr[...] = value
                except Exception:
                    try:
                        setattr(ret_rms, attr, np.asarray(arr) * 0.0 + value)
                    except Exception:
                        pass
        if hasattr(ret_rms, "count"):
            try:
                ret_rms.count = 1e-4
            except Exception:
                pass

    returns = getattr(vec_env, "returns", None)
    if returns is not None:
        try:
            returns[...] = 0.0
        except Exception:
            pass
    print(f"[INFO] {label}: reward normalization stats reset; observation normalization preserved.")



def _flatten_dict_for_diff(prefix: str, data: dict, out: dict[str, object]) -> None:
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_dict_for_diff(full_key, value, out)
        else:
            out[full_key] = value


def _print_hyperparam_diff(checkpoint_cfg: dict, live_cfg: dict, keys: list[str]) -> None:
    flat_ckpt: dict[str, object] = {}
    flat_live: dict[str, object] = {}
    _flatten_dict_for_diff("", checkpoint_cfg, flat_ckpt)
    _flatten_dict_for_diff("", live_cfg, flat_live)

    print("[VERIFY] PPO checkpoint-vs-live hyperparameter diff:")
    any_diff = False
    for key in keys:
        ckpt_val = flat_ckpt.get(key, "<missing>")
        live_val = flat_live.get(key, "<missing>")
        if ckpt_val != live_val:
            any_diff = True
            print(f"[VERIFY]   {key}: checkpoint={ckpt_val} | live={live_val}")
    if not any_diff:
        print("[VERIFY]   no diff detected for requested keys")


def setup_model(
    env,
    *,
    args_cli,
    agent_cfg,
    algo_cls,
    policy_class,
    actor_obs_dim,
    obs_dim,
    privileged_obs_dim,
    params_dir,
    log_dir,
):
    """Build (fresh) or resume the PPO/DivCPPO model. Returns (model, env)."""
    if args_cli.checkpoint is not None:
        checkpoint_dir = Path(args_cli.checkpoint).resolve().parent
        active_agent_cfg = dict(agent_cfg)
        active_agent_cfg["policy_kwargs"] = dict(agent_cfg.get("policy_kwargs", {}))
        resume_overrides_from_settings: dict[str, float | int | None] = {}

        resume_mapping = {
            "learning_rate": args_cli.resume_learning_rate,
            "clip_range": args_cli.resume_clip_range,
            "ent_coef": args_cli.resume_ent_coef,
            "vf_coef": args_cli.resume_vf_coef,
            "n_epochs": args_cli.resume_n_epochs,
            "batch_size": args_cli.resume_batch_size,
            "target_kl": args_cli.resume_target_kl,
        }
        for key, value in resume_mapping.items():
            if value is not None:
                if key in ("n_epochs", "batch_size"):
                    active_agent_cfg[key] = int(value)
                    resume_overrides_from_settings[key] = int(value)
                else:
                    active_agent_cfg[key] = float(value)
                    resume_overrides_from_settings[key] = float(value)

        runtime_log_std = args_cli.resume_log_std_init
        if runtime_log_std is None:
            runtime_log_std = active_agent_cfg.get("policy_kwargs", {}).get("log_std_init", None)
        if runtime_log_std is not None:
            active_agent_cfg["policy_kwargs"]["log_std_init"] = float(runtime_log_std)

        # Resume intentionally follows run_settings.py. The checkpoint provides
        # network weights and optimizer state; active PPO hyperparameters come
        # from PPO_DEFAULTS plus optional RESUME_DEFAULTS.
        dump_yaml(os.path.join(params_dir, "agent.yaml"), build_agent_snapshot_cfg(active_agent_cfg, args_cli=args_cli, actor_obs_dim=actor_obs_dim, obs_dim=obs_dim, privileged_obs_dim=privileged_obs_dim))

        print(f"[INFO] Loading checkpoint (PPO): {args_cli.checkpoint}")
        checkpoint_agent_cfg = _load_yaml_dict(str(checkpoint_dir / "params" / "agent.yaml"))
        if checkpoint_agent_cfg:
            live_compare_cfg = _stable_config_repr(build_agent_snapshot_cfg(active_agent_cfg, args_cli=args_cli, actor_obs_dim=actor_obs_dim, obs_dim=obs_dim, privileged_obs_dim=privileged_obs_dim))
            diff_keys = [
                "learning_rate",
                "n_steps",
                "batch_size",
                "n_epochs",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "vf_coef",
                "max_grad_norm",
                "target_kl",
                "ppo_head",
                "asymmetric_critic",
                "actor_obs_dim",
                "critic_obs_dim",
                "privileged_obs_dim",
                "policy_kwargs.net_arch",
                "policy_kwargs.actor_obs_dim",
                "policy_kwargs.log_std_init",
                "policy_kwargs.decomposed_config",
                "policy_kwargs.use_sde",
            ]
            _print_hyperparam_diff(checkpoint_agent_cfg, live_compare_cfg, diff_keys)

        # Compare the whole reward config between checkpoint and current env.
        # If it changed, keep checkpoint obs_rms (policy depends on it) and reset only reward stats.
        checkpoint_env_yaml = _load_yaml_dict(str(checkpoint_dir / "params" / "env.yaml"))
        live_env_yaml = _load_yaml_dict(os.path.join(params_dir, "env.yaml"))
        ck_rewards = _reward_config_signature(checkpoint_env_yaml)
        live_rewards = _reward_config_signature(live_env_yaml)
        reward_cfg_changed = ck_rewards != live_rewards
        if reward_cfg_changed:
            ck_terms = set(ck_rewards.keys())
            live_terms = set(live_rewards.keys())
            added = sorted(live_terms - ck_terms)
            removed = sorted(ck_terms - live_terms)
            changed = sorted(k for k in ck_terms & live_terms if ck_rewards.get(k) != live_rewards.get(k))
            print(
                "[INFO] Reward config changed vs checkpoint: "
                f"added={added or []}, removed={removed or []}, changed={changed or []}"
            )

        from evaluation.checkpoint_assets import resolve_vecnormalize_path
        vecnorm_path = resolve_vecnormalize_path(args_cli.checkpoint)
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True
        print(f"[INFO] VecNormalize loaded for resume (PPO): {vecnorm_path}")
        if reward_cfg_changed:
            _reset_vecnormalize_reward_stats(env, label="VecNormalize(after_reward_change)")

        # Persist the active VecNormalize stats for this run.
        try:
            if hasattr(env, "save"):
                env.save(os.path.join(params_dir, "vecnormalize.pkl"))
                print(f"[INFO] Active VecNormalize saved to {os.path.join(params_dir, 'vecnormalize.pkl')}")
        except Exception:
            print("[WARN] Failed to save active VecNormalize to params dir.")

        def _const_schedule(value: float):
            return lambda _: float(value)

        ppo_custom_objects = {
            "learning_rate": active_agent_cfg["learning_rate"],
            "n_steps": active_agent_cfg["n_steps"],
            "batch_size": active_agent_cfg["batch_size"],
            "n_epochs": active_agent_cfg["n_epochs"],
            "gamma": active_agent_cfg["gamma"],
            "gae_lambda": active_agent_cfg["gae_lambda"],
            "clip_range": active_agent_cfg["clip_range"],
            "ent_coef": active_agent_cfg["ent_coef"],
            "vf_coef": active_agent_cfg["vf_coef"],
            "max_grad_norm": active_agent_cfg["max_grad_norm"],
            "target_kl": active_agent_cfg.get("target_kl", None),
        }

        model = algo_cls.load(
            args_cli.checkpoint,
            env=env,
            verbose=1,
            device="cuda",
            tensorboard_log=log_dir,   # log into the new run dir, not the old checkpoint dir
            custom_objects=ppo_custom_objects,
        )
        apply_ppo_head_runtime(model, env, "resume", args_cli=args_cli, actor_obs_dim=actor_obs_dim, obs_dim=obs_dim, privileged_obs_dim=privileged_obs_dim)

        # SB3 restores optimizer state from checkpoint after setup; make sure the
        # active learning rate from run_settings.py also reaches optimizer groups.
        new_lr = float(active_agent_cfg["learning_rate"])
        model.learning_rate = new_lr
        model.lr_schedule = _const_schedule(new_lr)
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr

        model.n_steps = int(active_agent_cfg["n_steps"])
        model.batch_size = int(active_agent_cfg["batch_size"])
        model.n_epochs = int(active_agent_cfg["n_epochs"])
        model.gamma = float(active_agent_cfg["gamma"])
        model.gae_lambda = float(active_agent_cfg["gae_lambda"])
        model.clip_range = _const_schedule(float(active_agent_cfg["clip_range"]))
        model.ent_coef = float(active_agent_cfg["ent_coef"])
        model.vf_coef = float(active_agent_cfg["vf_coef"])
        model.max_grad_norm = float(active_agent_cfg["max_grad_norm"])
        model.target_kl = active_agent_cfg.get("target_kl", None)

        ppo_overrides = {
            "ppo_head": args_cli.ppo_head,
            "asymmetric_critic": bool(args_cli.asymmetric_critic),
            "actor_obs_dim": actor_obs_dim,
            "critic_obs_dim": obs_dim,
            "privileged_obs_dim": privileged_obs_dim,
            "learning_rate": new_lr,
            "n_steps": int(active_agent_cfg["n_steps"]),
            "batch_size": int(active_agent_cfg["batch_size"]),
            "n_epochs": int(active_agent_cfg["n_epochs"]),
            "gamma": float(active_agent_cfg["gamma"]),
            "gae_lambda": float(active_agent_cfg["gae_lambda"]),
            "clip_range": float(active_agent_cfg["clip_range"]),
            "ent_coef": float(active_agent_cfg["ent_coef"]),
            "vf_coef": float(active_agent_cfg["vf_coef"]),
            "max_grad_norm": float(active_agent_cfg["max_grad_norm"]),
            "target_kl": active_agent_cfg.get("target_kl", None),
        }
        if args_cli.decomposed_advantage:
            ppo_overrides["decomposed_config"] = active_agent_cfg.get("policy_kwargs", {}).get("decomposed_config", {})

        runtime_log_std = active_agent_cfg.get("policy_kwargs", {}).get("log_std_init", None)
        if runtime_log_std is not None:
            new_log_std = float(runtime_log_std)
            if hasattr(model, "policy") and hasattr(model.policy, "log_std"):
                with torch.no_grad():
                    model.policy.log_std.fill_(new_log_std)
                if hasattr(model.policy, "optimizer"):
                    model.policy.optimizer.state.pop(model.policy.log_std, None)
                ppo_overrides["policy.log_std"] = new_log_std
                ppo_overrides["policy.std"] = float(math.exp(new_log_std))
                print(
                    f"[INFO] PPO log_std from run_settings applied: log_std={new_log_std:.4f} "
                    f"(std={math.exp(new_log_std):.4f})"
                )
            else:
                print("[WARNING] PPO_DEFAULTS['log_std_init'] ignored: model.policy.log_std not found.")

        buffer_size = getattr(getattr(model, "rollout_buffer", None), "buffer_size", None)
        if buffer_size is not None and int(buffer_size) != int(model.n_steps):
            print(
                f"[WARNING] Rollout buffer size ({buffer_size}) != model.n_steps ({model.n_steps}). "
                "If this appears, restart from a checkpoint with matching PPO_DEFAULTS['n_steps']."
            )

        print(
            "[INFO] PPO resume hyperparameters applied from run_settings.py: "
            + ", ".join(f"{k}={v}" for k, v in ppo_overrides.items())
        )
        if resume_overrides_from_settings:
            print(
                "[INFO] RESUME_DEFAULTS overrides on top of PPO_DEFAULTS: "
                + ", ".join(f"{k}={v}" for k, v in resume_overrides_from_settings.items())
            )
        dump_yaml(os.path.join(params_dir, "resume_overrides.yaml"), ppo_overrides)
        _print_vecnormalize_state(env, label="VecNormalize(after_resume_load)")

    else:
        normalize_kwargs = dict(norm_obs=True, norm_reward=True, clip_obs=10.0,
                                gamma=float(agent_cfg.get("gamma", PPO_DEFAULTS["gamma"])))
        if args_cli.decomposed_advantage:
            env = VecNormalize(env, **normalize_kwargs)
        else:
            env = PpoRewardVecNormalize(
                env, reward_units=args_cli.scalar_reward_units,
                step_dt=env.unwrapped.step_dt, **normalize_kwargs,
            )
        print(f"[VERIFY] Post-wrap wrapper chain: {' -> '.join(_wrapper_chain(env))}")
        print(f"[VERIFY] Wrapped observation_space: {_space_summary(getattr(env, 'observation_space', None))}")
        print(f"[VERIFY] Wrapped action_space: {_space_summary(getattr(env, 'action_space', None))}")
        _print_vecnormalize_state(env, label="VecNormalize(initial)")
        print("[INFO] Creating a new PPO model")
        model = algo_cls(
            policy=policy_class,
            env=env,
            verbose=1,
            device="cuda",
            tensorboard_log=log_dir,
            **agent_cfg,
        )
        if not args_cli.decomposed_advantage:
            model.scalar_reward_units = env.scalar_reward_units
            model.reward_stats_units = "per_step"
        apply_ppo_head_runtime(model, env, "fresh", args_cli=args_cli, actor_obs_dim=actor_obs_dim, obs_dim=obs_dim, privileged_obs_dim=privileged_obs_dim)

    return model, env
