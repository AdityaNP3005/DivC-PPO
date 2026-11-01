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

- Linux, an NVIDIA GPU with a recent driver, Docker with the NVIDIA Container Toolkit.
- The `nvcr.io/nvidia/isaac-sim:5.1.0` image.
- A host directory `$ISAAC_HOST_ROOT/storage` that contains:
  - `IsaacLab/`: an Isaac Lab 2.3.2 checkout.
  - `venv/lib/python3.11/site-packages/`: the packages in `requirements.txt`.

  The container mounts this directory read-only at `/storage`. One way to install the packages
  with the Isaac Sim Python:

  ```bash
  export ISAAC_HOST_ROOT=/path/to/isaac_root
  docker run --rm --gpus all --entrypoint bash -e ACCEPT_EULA=Y \
      -v "$ISAAC_HOST_ROOT/storage:/storage" -v "$PWD:/divc_ppo" nvcr.io/nvidia/isaac-sim:5.1.0 \
      -c '/isaac-sim/python.sh -m pip install --target /storage/venv/lib/python3.11/site-packages -r /divc_ppo/requirements.txt'
  ```

The paper runs used Python 3.11.13, PyTorch 2.7.0+cu128 (bundled with Isaac Sim),
Stable-Baselines3 2.7.1, Gymnasium 1.2.3 and NumPy 1.26.0. The scene, the cylinder and the
fine-tuned YOLO11n detector are in `assets/`.

## Quick start

All commands run from the repository root on the host. `isaac-ctl.sh` starts one long-lived
container (`./isaac-ctl.sh start --runtime-only`, `./isaac-ctl.sh stop`), and `run.sh` launches
jobs in it. `GPU=<index>` selects the GPU, and the default is set in `config/training.py`.

```bash
export ISAAC_HOST_ROOT=/path/to/isaac_root

./run.sh train divc-ppo --seed 42 --headless          # train, then play videos and a nominal validation
./run.sh train ppo --seed 42 --headless
./run.sh train divc-ppo --control shared-ratio --seed 42 --headless   # Table IV controls
./run.sh train divc-ppo --control scrambled --seed 42 --headless
./run.sh train divc-ppo --dry-run                     # print the environment of a preset

./run.sh validate logs/sb3/Isaac-UR10e-DivC-Reach-v0/<run>/model.zip --condition ood_r0.14 --seed 42
./run.sh play logs/sb3/Isaac-UR10e-DivC-Reach-v0/<run>/model.zip
```

Each training run takes 1M environment steps with 128 parallel environments, about 72 minutes
on our hardware. Models are written to `logs/sb3/Isaac-UR10e-DivC-Reach-v0/<timestamp>_.../`,
and videos and validations to `play_outputs/`.

## Reproducing the paper

```bash
python3 tools/reproduce.py --dry-run    # list the 12 trainings and 108 evaluations
python3 tools/reproduce.py              # run them; finished steps are skipped on restart
python3 tools/tables.py results         # Tables III-V -> results/tables.md
python3 tools/figures.py results        # Figs. 3-5 -> results/*.pdf
```

`config/experiment.py` defines the presets:

| Name | Paper | Setting |
|---|---|---|
| `ppo` | PPO baseline | joint advantage and ratio, summed reward |
| `divc-ppo` | DivC-PPO | per-axis reward streams, advantages and ratios |
| `--control shared-ratio` | Table IV | per-axis rewards and advantages, one joint ratio |
| `--control scrambled` | Table IV | X stream receives the Y reward, Y the Z reward, Z the X reward |
| `nominal` | Table III, Fig. 3 | object within ±4 cm, 1,024 episodes |
| `noise_<σ>` | Table V, Fig. 5a | Gaussian noise on the bounding-box center, σ × 640 px |
| `ood_r<r>` | Table V, Fig. 5b | object at radius r (m) from the nominal point |
| `scale_<s>` | Table V, Fig. 5c | action scale s (m) instead of 0.015 |

Both methods use the same symmetric critic, reward terms (`config/rewards_weights.py`,
[docs/REWARD_DESIGN.md](docs/REWARD_DESIGN.md)) and PPO hyperparameters (`config/training.py`).
Isaac Sim physics and rendering are not bitwise deterministic, so retrained models differ
slightly from ours. Expect results within the spread across seeds reported in the paper.

## Repository layout

```
config/           presets (experiment.py), task, reward weights, PPO and camera settings
rl_env/env/       Isaac Lab scene, observations, rewards, events
rl_env/policy/    DivC-PPO learner (divc_ppo.py), per-axis reward split (axis_rewards.py)
rl_env/control/   Cartesian action and damped-least-squares IK
training/         model construction, normalization, callbacks
evaluation/       deterministic validation and its metrics
tools/            reproduce.py, tables.py, figures.py
train_ur10e.py    training entry point
validate_inference.py, play.py   evaluation and videos
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
