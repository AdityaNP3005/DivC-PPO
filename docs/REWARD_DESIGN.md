# Reward design

## Path

```
config/rewards_weights.py   weights and scales        config/reward_units.reward_error_source() -> "gt" | "vision"
            |                                                    |  (runtime.REWARD_ERROR_SOURCE)
            v                                                    v
rl_env/env/rewards_cfg.py   one line per term: _target_term(func, weight, axis/scales...), error_source in params
            v
rl_env/rewards/target_terms.py, safety_terms.py
            v
error_sources._reward_error(..., error_source) -> _gt_ee_target_error_w (paper) | _vision_error
            v
IsaacLab RewardManager: sum of func * weight * step_dt  ->  learner (rl_env/policy/axis_rewards.py)
```

PPO and DivC-PPO use the same terms, weights and simulator-state (GT) error. The presets in
`config/experiment.py` change only the credit assignment. PPO receives the summed reward.
DivC-PPO splits it into X, Y and Z streams by term name (`rl_env/policy/axis_rewards.py`), and
task-wide terms are copied to all three streams. The vision error source is kept as an option
and is not used in the paper. With it, `low_depth` is removed because that term reads the GT
object top (`rl_env/env/assembly.py`).

## Coarse and fine exponential terms

`exp_*_coarse` and `exp_*` (fine) are two reward terms on the same axis. Both evaluate
`exp(-|e| / scale)` with different scales and are summed on every step. The fine term is close
to zero far from the target, so no switch is needed.

| Term | Weight X/Y/Z | Scale XY / Z | Role |
|---|---|---|---|
| `exp_*_coarse` | 15 / 15 / 60 | 0.15 m / 0.70 m | pull from the far field (XY up to about 15 cm, the ~0.5 m Z descent) |
| `exp_*` (fine) | 40 / 40 / 80 | 0.008 m / 0.10 m | steep gradient in the last ~3 cm (XY) for sub-millimeter centering |

Reward gained per cm of approach (weight x slope), XY:

| Distance | 0.2 cm | 0.8 cm | 3 cm | 10 cm | 15 cm | 30 cm |
|---|---|---|---|---|---|---|
| progress | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| abs | 0.67 | 0.67 | 0.67 | 0.67 | 0 | 0 |
| coarse | 0.99 | 0.95 | 0.82 | 0.51 | 0.37 | 0.14 |
| fine | 38.9 | 18.4 | 1.2 | 0 | 0 | 0 |

The fine XY scale of 0.008 m matters. A narrower scale (0.005 m) starved the mid-range
approach in earlier tuning runs.

## Known redundancy

These terms are kept unchanged so that the code matches the paper checkpoints. Changing any of
them changes the reward and requires retraining both methods.

- `abs_x/y` and `exp_x/y_coarse` have similar slopes (0.4–1.0 per cm) over the same 0–15 cm range.
- `abs_z` (0.17 per cm) is small next to progress Z (1.33) and coarse Z (~0.7).
- `alive_penalty` and `stay_xy` use the same in-band indicator (the 1 mm target box). With
  training episodes that end only at the time limit, the constant offset does not change the
  optimal policy.
- `target_stay_reward` returns `in_band * step_dt`, and the RewardManager multiplies by `step_dt`
  again. The effective stay reward is therefore weight * dt^2 per step (about 1.0 for
  W_STAY = 30 at 30 Hz).
