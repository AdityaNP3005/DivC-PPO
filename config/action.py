from __future__ import annotations

"""Tunable Cartesian action / IK knobs.

TUNABLE — edit values here. The fixed IK implementation (CartesianDeltaAction,
6-DOF orientation-hold DLS solve) reads these from `config`.
"""

from .env_flags import env_scale3

# Cartesian action controller settings. Action 1.0 means this many meters before
# IK/joint-speed limits. With 0.015 m, action 1.0 requests 1.5 cm.
# `position_scale_m` is env-overridable (POSITION_SCALE_M) so the action-scale
# sweep can retrain variant-4 at 0.015→0.060 without editing this file; unset → default.
CARTESIAN_ACTION_DEFAULTS = {
    "position_scale_m": env_scale3("POSITION_SCALE_M", (0.015, 0.015, 0.015)),
    # 6-DOF damped-least-squares IK regularization (λ²I in J Jᵀ + λ²I).
    "ik_damping": 0.05,
    # Per-step joint speed limit (deg/s); the WHOLE delta_q is scaled uniformly.
    "max_joint_speed_deg_s": 120.0,
    # k_ori multiplier on the Siciliano orientation error in the IK task vector.
    "orientation_gain": 1.0,
}
