"""Paper presets: the two methods, the two mechanism controls, and the evaluation conditions.

Both methods use the symmetric critic and the simulator-state (GT) training reward, as in the
paper. Each preset is a bundle of environment variables that run.sh forwards into the container.

    python -m config.experiment method divc-ppo [--control shared-ratio]
    python -m config.experiment condition noise_0.0005
"""

PROTOCOL = "divc_ppo_paper"
SEEDS = (42, 44, 46)

COMMON_ENV = {
    "RL_CONTROL_MODE": "cartesian", "PPO_HEAD": "single", "ASYMMETRIC_CRITIC": "0",
    "SCALAR_CRITIC": "1", "REWARD_ERROR_SOURCE": "gt",
    "DIVC_REWARD_UNITS": "legacy_rate", "PPO_REWARD_UNITS": "legacy_rate",
    "DECOMPOSED_AXIS_WEIGHTS": "1,1,1", "SHARED_RATIO": "0", "AXIS_CREDIT_PERMUTATION": "none",
    "ABLATION_PROTOCOL": PROTOCOL,
}
# name -> (arm id stored in checkpoint metadata, per-axis credit)
METHODS = {"ppo": (1, False), "divc-ppo": (3, True)}
# Mechanism controls (Table IV), applied on top of divc-ppo.
CONTROLS = {
    "shared-ratio": {"SHARED_RATIO": "1"},              # per-axis rewards/GAE, one joint ratio
    "scrambled": {"AXIS_CREDIT_PERMUTATION": "cyclic"},  # X stream gets the Y reward, and so on
}

NOMINAL_EVAL = {"EVAL_NUM_ENVS": "128", "EVAL_MIN_EPISODES": "1000"}
BATTERY_EVAL = {"EVAL_NUM_ENVS": "100", "EVAL_MIN_EPISODES": "100"}
# Deployment perturbations (Table V, Fig. 5); the policies are not retrained.
NOISE_STD = ("0.0001", "0.00025", "0.0005", "0.001", "0.0015", "0.0025")  # x 640 px = 0.064-1.6 px
RADII_M = ("0.06", "0.08", "0.10", "0.12", "0.14")
ACTION_SCALES_M = ("0.020", "0.025", "0.030", "0.035", "0.040")
CONDITIONS = {"nominal": {}}
CONDITIONS.update({f"noise_{s}": {"YOLO_OBS_NOISE_STD": s} for s in NOISE_STD})
CONDITIONS.update({f"ood_r{r}": {"PLAY_OBJECT_RADIUS_M": r} for r in RADII_M})
CONDITIONS.update({f"scale_{a}": {"POSITION_SCALE_M": a} for a in ACTION_SCALES_M})


def method_env(method, control=None):
    arm, per_axis = METHODS[method]
    if control and method != "divc-ppo":
        raise ValueError("Mechanism controls apply to divc-ppo only")
    flag = "1" if per_axis else "0"
    env = dict(COMMON_ENV, ABLATION10_ARM=str(arm), DECOMPOSED_ADVANTAGE=flag, SINGLE_DECOMPOSED_ADVANTAGE=flag)
    env.update(CONTROLS[control] if control else {})
    return env


def condition_env(condition):
    base = NOMINAL_EVAL if condition == "nominal" else BATTERY_EVAL
    return dict(base, ABLATION_EVAL_CONDITION=condition, **CONDITIONS[condition])


def _main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("method")
    m.add_argument("name", choices=tuple(METHODS))
    m.add_argument("--control", choices=tuple(CONTROLS))
    c = sub.add_parser("condition")
    c.add_argument("name", choices=tuple(CONDITIONS))
    args = parser.parse_args(argv)
    env = method_env(args.name, args.control) if args.kind == "method" else condition_env(args.name)
    for key, value in env.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
