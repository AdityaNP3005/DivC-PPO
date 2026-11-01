"""Training orchestration for single-actor PPO and DivC-PPO (entrypoint: train_ur10e.py).

    runtime_args         — pre-Kit: resolve runtime args, stdout/stderr noise filter
    camera_probe         — one-shot wrist camera + YOLO gate before training
    model_builder        — single-actor/asymmetric PPO and DivC-PPO wiring
    resume               — fresh model or checkpoint resume, plus resume diagnostics
    run_artifacts        — run-dir name, periodic checkpoints, training_result.json handoff
    reward_normalization — scalar PPO reward-rate VecNormalize
    callbacks/           — SB3 callbacks (monitor, per-term reward logger, axis-credit probe)

Isaac-coupled modules are imported only after AppLauncher has started.
"""
