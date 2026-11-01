# DivC-PPO

Code for the paper

> **DivC-PPO: Cartesian-Aligned Credit Assignment for Precision Vision-Guided End-Effector Positioning**
> Aditya Nugraha Putra, Van-Linh Nguyen. National Chung Cheng University.
> Project page: <https://divcppo.vercel.app>

DivC-PPO gives each Cartesian axis its own reward stream, advantage and clipped probability ratio
inside PPO. One critic value is the baseline for every stream, and the per-axis clipped surrogates
are summed into the policy loss. The task is UR10e end-effector positioning in Isaac Lab:
- The actor and critic observe only a 12-D vector of wrist-camera YOLO11n detection features and proprioception.
- The action is a Cartesian XYZ increment, clipped to [-1, 1], scaled by 1.5 cm, and converted to joint targets by damped-least-squares inverse kinematics.
- Training rewards are computed from the simulator state.

| Nominal evaluation (3 seeds, 1,024 episodes each) | PPO | DivC-PPO |
|---|---:|---:|
| 3-D error (mm) | 1.167 ± 0.103 | 0.337 ± 0.059 |
| Stable success (%) | 56.64 ± 5.53 | 100.00 ± 0.00 |
| Near-target failure (%) | 41.15 ± 5.22 | 0.00 ± 0.00 |
| Time to stable success (s) | 2.14 ± 0.06 | 1.55 ± 0.03 |

## Requirements

- Isaac Sim 5.1.0 and Isaac Lab 2.3.2.
- The Python packages in `requirements.txt`, installed into the Isaac Lab Python environment:
  ```bash
  isaaclab.sh -p -m pip install -r requirements.txt
  ```

The paper runs used Python 3.11.13, PyTorch 2.7.0+cu128 (bundled with Isaac Sim),
Stable-Baselines3 2.7.1, Gymnasium 1.2.3 and NumPy 1.26.0. The scene, the cylinder and the
fine-tuned YOLO11n detector are in `assets/`.

## Training and evaluation

Run from the repository root. Here `isaaclab.sh` is the Isaac Lab launcher (`./isaaclab.sh` in the
Isaac Lab checkout). A preset from `config/experiment.py` is exported as environment variables
before the run. `GPU=<index>` selects the GPU (default 0).

```bash
export $(python3 -m config.experiment method divc-ppo)        # or: method ppo
SEED=42 isaaclab.sh -p train_ur10e.py --headless

export $(python3 -m config.experiment condition ood_r0.14)    # or: nominal, noise_0.0005, scale_0.030, ...
isaaclab.sh -p evaluate.py --checkpoint logs/sb3/Isaac-UR10e-DivC-Reach-v0/<run>/model.zip \
    --seed 42 --out results/eval/divc-ppo_seed42/ood_r0.14
```

A training run requests 1M environment steps over 128 parallel environments. It completes
after 31 rollouts of 256 steps (1,015,808 steps), which took about 72 minutes on our hardware.
Models and TensorBoard logs are written to `logs/sb3/Isaac-UR10e-DivC-Reach-v0/<timestamp>_.../`.
The validator writes `inference_metrics.json` and `inference_records.npz` to `--out`.

## Reproducing the paper

```bash
python3 tools/reproduce.py --dry-run    # list the 12 trainings and 108 evaluations
python3 tools/reproduce.py              # run them; finished steps are skipped on restart
python3 tools/tables.py results         # Tables III-V -> results/tables.md
python3 tools/figures.py results        # Figs. 3-5 -> results/*.pdf
```

`tools/reproduce.py` starts each run with `$ISAACLAB_SH -p` (default: `isaaclab.sh` on `PATH`).
`config/experiment.py` defines the presets:

| Name | Paper | Setting |
|---|---|---|
| `method ppo` | PPO baseline | joint advantage and ratio, summed reward |
| `method divc-ppo` | DivC-PPO | per-axis reward streams, advantages and ratios |
| `--control shared-ratio` | Table IV | per-axis rewards and advantages, one joint ratio |
| `--control scrambled` | Table IV | X stream receives the Y reward, Y the Z reward, Z the X reward |
| `condition nominal` | Table III, Fig. 3 | object within ±4 cm, 1,024 episodes |
| `condition noise_<σ>` | Table V, Fig. 5a | Gaussian noise on the bounding-box center, σ × 640 px |
| `condition ood_r<r>` | Table V, Fig. 5b | object at radius r (m) from the nominal point |
| `condition scale_<s>` | Table V, Fig. 5c | action scale s (m) instead of 0.015 |

Both methods use the same symmetric critic, reward terms (`config/rewards_weights.py`,
[docs/REWARD_DESIGN.md](docs/REWARD_DESIGN.md)) and PPO hyperparameters (`config/training.py`).
Isaac Sim physics and rendering are not bitwise deterministic, so retrained models differ
slightly from ours. Expect results within the spread across seeds reported in the paper.

## Repository layout

```
config/            presets (experiment.py), task, reward weights, PPO and camera settings
rl_env/env/        Isaac Lab scene, observations, rewards, events
rl_env/perception/ YOLO detection cache and the observation terms built from it
rl_env/policy/     DivC-PPO learner (divc_ppo.py), per-axis reward split (axis_rewards.py)
rl_env/control/    Cartesian action, damped-least-squares IK, object placement
training/          model construction, reward normalization, callbacks
evaluation/        deterministic validation and its metrics
tools/             reproduce.py, tables.py, figures.py
train_ur10e.py     training entry point
evaluate.py        evaluation entry point
```

## Citation

```bibtex
@inproceedings{putra2026divcppo,
  title  = {DivC-PPO: Cartesian-Aligned Credit Assignment for
            Precision Vision-Guided End-Effector Positioning},
  author = {Putra, Aditya Nugraha and Nguyen, Van-Linh},
  year   = {2026},
  note   = {To appear}
}
```

## License

MIT, see [LICENSE](LICENSE).
