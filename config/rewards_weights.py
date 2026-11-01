from __future__ import annotations

"""Tunable reward weights and shaping scales.

`rl_env/env/rewards_cfg.py` wires each RewardsCfg term to these names; the shaping
math lives in `rl_env/rewards/`. Sign convention: positive = bonus, negative = penalty.
IsaacLab's RewardManager multiplies every term by weight * step_dt.
See docs/REWARD_DESIGN.md for why each shaping term exists.
"""

from .env_flags import env_float

# ── Per-axis distance shaping (terms penalty_*, abs_*, exp_*_coarse, exp_*) ──────
W_PROGRESS_X = 5.0
W_PROGRESS_Y = 5.0
W_PROGRESS_Z = 20.0  # boosted: XY exp rewards otherwise dominate and the ~0.5 m Z descent stalls
W_ABS_X = -10.0
W_ABS_Y = -10.0
W_ABS_Z = -10.0
W_EXP_COARSE_X = 15.0
W_EXP_COARSE_Y = 15.0
W_EXP_COARSE_Z = 60.0
W_EXP_FINE_X = 40.0
W_EXP_FINE_Y = 40.0
W_EXP_FINE_Z = 80.0

# Error normalisation for progress + abs terms: |e| / scale (abs saturates at 1).
ERROR_NORM_XY_SCALE_M = 0.15
ERROR_NORM_Z_SCALE_M = 0.15
ABS_Z_SCALE_M = 0.6  # 0.15 pinned abs_z at its floor for the ~0.5 m descent (no gradient)

# Exponential exp(-|e| / scale). Coarse pulls from the far field; fine owns the last ~3 cm.
EXP_COARSE_XY_SCALE_M = 0.15
EXP_COARSE_Z_SCALE_M = 0.70  # reaches the full ~0.5 m reset→target descent
# 0.008 is the proven sub-mm value (run 2026-07-02_18-50-28: E_xy 0.494 mm, SR 1.0).
# DO-NOT-REPEAT: 0.005 (2026-07-03) regressed to E_xy 5.30 mm, SR 0.645 (mid-range starved).
# The env var keeps its historical name (an earlier action-scale sweep set 0.05).
EXP_FINE_XY_SCALE_M = env_float("GT_FINE_EXP_XY_SCALE_M", 0.008)
EXP_FINE_Z_SCALE_M = 0.10

# ── Target band, idle and safety ──────────────────────────────────────────────
W_SUCCESS_ONCE = 40.0
W_STAY = 30.0
W_ALIVE = -0.3  # light time pressure
W_IDLE = -5.0
IDLE_ACTION_THRESHOLD = 0.001  # |action| below this counts as idle
W_LOW_CLEARANCE = -2.0  # GT-only; removed in pure-vision runs
W_NO_DET = -5.0  # -10 was too aggressive during early exploration
W_ACTION_RATE = -1.5
W_JOINT_CLAMP = -20.0
JOINT_MAX_DEVIATION_RAD = 1.8
