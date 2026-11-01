from __future__ import annotations

"""Single home for environment-variable parsing + device resolution.

`run_ablation.sh` selects ablation variants by exporting env vars
(ASYMMETRIC_CRITIC / DECOMPOSED_ADVANTAGE /
RL_CONTROL_MODE / SEED). Historically the truthy parsing was duplicated in
train_ur10e.py and env_cfg.py; this module is the one place that logic should
live so the two can never drift.
"""

import os

# Strings that count as "true" for boolean env flags (case-insensitive).
_TRUTHY = ("1", "true", "yes", "on")


def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var with the project's standard truthy set.

    Empty/whitespace → default (so a shell wrapper can forward `${VAR:-}`
    unconditionally without an unset knob flipping the flag to False).
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def env_str(name: str, default: str) -> str:
    """Read a string env var, falling back to `default` when unset/empty."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to `default` when unset/empty/unparsable.

    Tolerating empty strings lets shell wrappers forward `${VAR:-}` unconditionally
    (an unset knob → empty string → default) without changing behavior.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to `default` when unset/empty/unparsable.

    Accepts float-like strings (e.g. "2e6", "2000000.0") and truncates, so the
    sweep driver can pass MAX_STEPS in scientific notation.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        return default


def env_scale3(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Read a 3-axis scale env var → (x, y, z) tuple, falling back to `default`.

    Accepts either a single float (broadcast to all 3 axes, e.g. POSITION_SCALE_M=0.03)
    or 3 comma/space-separated floats. Unset/empty/unparsable → `default`.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    parts = raw.replace(",", " ").split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return default
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2])
    return default


def resolve_yolo_device(yolo_device: str | None, gpu: int | None, run_device: str | None = None) -> str:
    """Resolve YOLO device after CUDA_VISIBLE_DEVICES masking.

    Isaac/torch see the masked physical GPU as local cuda:0. This keeps
    `--gpu 1 --yolo_device 1` from accidentally requesting a non-existent local
    cuda:1 after masking.
    """

    text = "auto" if yolo_device is None else str(yolo_device).strip()
    if not text or text.lower() == "auto":
        if run_device and str(run_device).startswith("cuda:"):
            return str(run_device).split(":", 1)[1]
        return "0" if gpu is not None else "cpu"

    if gpu is not None and text == str(gpu):
        return "0"

    return text


def expose_gpu(args, run_device: str, *, flush: bool = False) -> None:
    """Expose one physical GPU (args.gpu) to this process and remap the YOLO device to it.

    Must run before CUDA initializes; the process then sees that GPU as local cuda:0.
    """
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    raw_yolo_device = str(args.yolo_device)
    args.yolo_device = resolve_yolo_device(args.yolo_device, args.gpu, run_device)
    if args.yolo_device != raw_yolo_device:
        print(
            f"[YOLO] device remapped: {raw_yolo_device} -> {args.yolo_device} "
            f"because --gpu {args.gpu} is exposed inside the process as {run_device}",
            flush=flush,
        )
