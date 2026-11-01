"""Rebuild the saved environment/model before deterministic inference."""


def setup():
    """
    Validasi INFERENCE deterministik paralel — UR10e centering.

    Runs a checkpoint deterministically (no exploration noise) in many parallel envs
    until at least --min_episodes episodes finish, then measures the GT error distribution
    (E_xy / E_z / norm), stable success (20 step berturut-turut), precision-hold,
    and splits bias from jitter. One-step reach is also logged as a diagnostic.

    This complements the training metrics (stochastic rollouts, from TensorBoard) with
    deterministic inference metrics, i.e. the precision of the deployed policy.

    Rumus GT (target/EE/error, toleransi sukses) meniru PERSIS
    `training/callbacks/monitor_stats.py::_accumulate_gt_stats` so both are comparable.

    Run from the host:
        ./run.sh validate --checkpoint /divc_ppo/logs/sb3/.../model.zip --num_envs 128 --min_episodes 1000

    Output → <play_outputs>/<checkpoint_tag>_inference_val_<ts>/:
        inference_metrics.json   ← statistik lengkap (mean/std/median/p90/p95, bias/jitter)
        validation_summary.txt   ← tabel 2 metrik bersanding (Training vs Inference)
        hist_exy.png             ← histogram sebaran E_xy (bias vs scatter secara visual)
    """

    # ─── 1. Parse argumen SEBELUM import Isaac Sim ────────────────────────────────
    import argparse
    import os
    import sys

    from isaaclab.app import AppLauncher

    _SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)

    # Import rl_env from this project, not from shared storage
    for _m in list(sys.modules):
        if _m == "rl_env" or _m.startswith("rl_env."):
            del sys.modules[_m]

    from rl_env.project_config import PLAY_ENV_ID, log_base

    from config import PLAY_DEFAULTS, TRAIN_DEFAULTS
    from evaluation.inference_bootstrap import (
        apply_runtime_defaults, load_checkpoint_context, apply_checkpoint_model,
        configure_inference_device, configure_inference_detector, load_inference_policy,
    )

    from evaluation.validation_contract import (
        positive_int, nonnegative_int, evaluation_context, create_evaluation_directory, atomic_json,
    )
    from config.training import VALIDATION_DEFAULTS
    from evaluation.checkpoint_contract import METHODS, require_normalizer_method

    parser = argparse.ArgumentParser(description="Validasi inference deterministik paralel UR10e.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override the PLAY_DEFAULTS checkpoint; if both are empty, use the latest model.zip.")
    parser.add_argument("--seed", type=nonnegative_int, default=None, help="Explicit evaluation seed.")
    parser.add_argument("--method", choices=tuple(METHODS), default=None, help="Require the matching new-protocol baseline.")
    parser.add_argument("--num_envs", type=positive_int, default=positive_int(VALIDATION_DEFAULTS["num_envs"]),
                        help="Number of parallel envs (default 128, as in training).")
    parser.add_argument("--min_episodes", type=positive_int, default=positive_int(VALIDATION_DEFAULTS["min_episodes"]),
                        help="Stop after at least N finished episodes (default 1000).")
    parser.add_argument("--max_steps", type=nonnegative_int, default=nonnegative_int(VALIDATION_DEFAULTS["max_steps"]),
                        help="Step limit (0 = auto: ceil(min_ep/num_envs+2)*280).")
    parser.add_argument("--out", type=str, default=None,
                        help="Folder output. Kosong → <play_outputs>/<tag>_inference_val_<ts>/.")
    AppLauncher.add_app_launcher_args(parser)

    # Cameras are required: the policy observes YOLO detections from the wrist camera.
    os.environ.setdefault("ENABLE_CAMERAS", "1")
    os.environ.setdefault("HEADLESS", "1")
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")

    args_cli = parser.parse_args()

    # ── Defaults from PLAY_DEFAULTS (device, YOLO, camera), as in play.py ──
    from config.env_flags import env_int
    args_cli.seed = args_cli.seed if args_cli.seed is not None else env_int("SEED", int(PLAY_DEFAULTS["seed"]))
    args_cli.seed = nonnegative_int(args_cli.seed)
    apply_runtime_defaults(args_cli, PLAY_DEFAULTS)
    print("[SETTINGS] Validation defaults loaded from config/training.py", flush=True)

    # Reuse the path restored by config.training before perception/task froze.
    args_cli.checkpoint = PLAY_DEFAULTS["checkpoint"]
    _candidate_checkpoint = args_cli.checkpoint
    if _candidate_checkpoint is None:
        print(f"[ERROR] No model.zip in {log_base()}.", flush=True)
        sys.exit(1)

    _checkpoint_context = load_checkpoint_context(
        _candidate_checkpoint, matched_validation=True, method=args_cli.method)
    _checkpoint_agent_cfg_early = _checkpoint_context.agent
    _matched_method = _checkpoint_context.method
    vecnorm_path = _checkpoint_context.normalizer
    apply_checkpoint_model(args_cli, _checkpoint_context, TRAIN_DEFAULTS)
    _RUN_DEVICE = configure_inference_device(args_cli)
    print(
        f"[MODE] control_mode={args_cli.mode} | PPO_head={args_cli.ppo_head} | "
        f"asym_critic={args_cli.asymmetric_critic} | "
        f"actor_depth_enabled={args_cli.actor_depth_enabled}",
        flush=True,
    )

    configure_inference_detector(args_cli)

    checkpoint_path = _candidate_checkpoint
    _validation_context = evaluation_context(checkpoint_path, args_cli.seed, args_cli.num_envs)
    _validation_context["comparison_protocol"] = _checkpoint_agent_cfg_early.get("comparison_protocol", "historical")
    _validation_context["method"] = _matched_method or "historical"
    from config.reward_units import reward_error_source
    _validation_context["reward_error_source"] = reward_error_source(args_cli.asymmetric_critic)
    out_dir = create_evaluation_directory(checkpoint_path, args_cli.out)
    _validation_context['eval_id'] = os.path.basename(out_dir)
    atomic_json(os.path.join(out_dir, "validation_status.json"), {
        **_validation_context, "validation_complete": False, "reason": "pending",
        "episodes_requested": args_cli.min_episodes, "episodes_completed": 0,
    })

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ─── 2. Imports after Isaac Sim starts ────────────────────────────────────────
    import rl_env  # noqa: F401
    from rl_env.env import PLAY_OBJECT_FARTHEST_RADIUS_M, UR10eEnvCfg
    from rl_env.env.factory import make_isaac_env

    ENV_ID = PLAY_ENV_ID

    # ─── 3. Output dir ───────────────────────────────────────────────────────────
    checkpoint_tag = os.path.basename(os.path.dirname(checkpoint_path))

    print(f"[INFO] Checkpoint : {checkpoint_path}", flush=True)
    print(f"[INFO] Output dir : {out_dir}", flush=True)

    # ─── 4. Build env (mirror train_ur10e.py, num_envs paralel) ──────────────────
    env_cfg = UR10eEnvCfg()

    # OOD placement (`ood_r*` conditions): push the object to a FIXED radius, random direction.
    # This is the TRAIN cfg (not UR10eEnvCfg_PLAY, which forces num_envs=1 + scripted grasp), so
    # the radial_offset_m that assembly.py sets on the PLAY cfg never reached us — the whole
    # ood_* family silently measured the nominal ±0.04 square. ponytail: applied ONLY when the
    # env var is explicitly set; the constant's default (0.04) would switch every nominal run
    # from uniform-square to fixed-radius ring and break comparability with past results.
    if os.environ.get("PLAY_OBJECT_RADIUS_M", "").strip():
        env_cfg.events.place_cylinder_under_gripper.params["radial_offset_m"] = PLAY_OBJECT_FARTHEST_RADIUS_M
        print(f"[OOD] object placement radius = {PLAY_OBJECT_FARTHEST_RADIUS_M:.4f} m "
              f"(fixed radius, random direction)", flush=True)

    base_env = make_isaac_env(
        env_cfg, ENV_ID, num_envs=args_cli.num_envs, seed=args_cli.seed, device=_RUN_DEVICE,
        camera=(args_cli.camera_resolution, args_cli.camera_update_period, "Validator"),
    )
    isaac_env = base_env.unwrapped

    _model_device = "cuda" if _RUN_DEVICE.startswith("cuda") else "cpu"
    env, model = load_inference_policy(
        base_env, checkpoint_path, vecnorm_path,
        seed=args_cli.seed, device=_model_device, simulation_app=simulation_app,
    )
    require_normalizer_method(env, model, _matched_method, isaac_env.step_dt)
    _validation_context["normalizer_class"] = type(env).__name__
    _validation_context["value_head_outputs"] = int(model.policy.value_net.out_features)
    print("[INFO] Single-actor model loaded", flush=True)

    # Runtime objects only; run() imports its own modules/constants.
    return {
        '_validation_context': _validation_context,
        'args_cli': args_cli,
        'checkpoint_path': checkpoint_path,
        'checkpoint_tag': checkpoint_tag,
        'env': env,
        'env_cfg': env_cfg,
        'isaac_env': isaac_env,
        'model': model,
        'out_dir': out_dir,
        'simulation_app': simulation_app,
    }
