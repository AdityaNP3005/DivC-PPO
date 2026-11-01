"""Everything a training run writes to its run dir: name, periodic checkpoints, final handoff.

training_result.json is the exact final-model handoff and source of identity for
train → play → validation.
"""
import os
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

from evaluation.validation_contract import atomic_json, file_digest


def run_dir_name(args_cli, timestamp: str) -> str:
    """Run directory name; tags mark only deviations from the matched-baseline defaults.

    Run discovery reads agent.yaml (analysis/run_index.py), never this tag.
    """
    from config.policy import CREDIT_DEFAULTS

    tag = f"{args_cli.ppo_head}_{'asym' if args_cli.asymmetric_critic else 'sym'}"
    if args_cli.ppo_head == "single" and args_cli.decomposed_advantage:
        tag += "_decomposed"
    if args_cli.decomposed_advantage and not getattr(args_cli, "scalar_critic", True):
        tag += "_axisvf"  # 3-dim per-axis value head (older runs: `_sharedvf`)
    if args_cli.decomposed_advantage:
        axis_w = tuple(CREDIT_DEFAULTS.get("decomposed_axis_weights", (1.0, 1.0, 1.0)))
        if axis_w != (1.0, 1.0, 1.0):
            tag += "_zw" + "-".join(f"{w:g}" for w in axis_w)  # non-unit axis weights
        if CREDIT_DEFAULTS["shared_ratio"]:
            tag += "_sharedratio"  # shared-ratio control (Table IV)
        perm = CREDIT_DEFAULTS["axis_credit_permutation"]
        if perm not in ("", "none", "identity", "0"):
            tag += f"_perm{perm}"
    if not args_cli.actor_depth_enabled:
        tag += "_xyonly"
    return f"{timestamp}_PPO_{tag}_seed{args_cli.seed}_policy"


class SmartCheckpointCallback(CheckpointCallback):
    def __init__(self, keep_last_n: int, **kwargs):
        if int(keep_last_n) < 1:
            raise ValueError("keep_last_n must be positive")
        super().__init__(**kwargs)
        self._keep_last_n = int(keep_last_n)
        self._saved_pairs = deque()
        self._current_paths = []

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        kind = checkpoint_type.strip("_.")
        suffix = "_" + kind if kind else ""
        extension = extension.lstrip(".")
        name = f"{self.name_prefix}_{self.num_timesteps}_steps{suffix}"
        if extension:
            name += "." + extension
        path = Path(self.save_path) / name
        if path.exists():
            raise FileExistsError(f"Preserving existing checkpoint artifact: {path}")
        self._current_paths.append(path)
        return str(path)

    def _on_step(self) -> bool:
        self._current_paths = []
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            self._saved_pairs.append(tuple(self._current_paths))
            while len(self._saved_pairs) > self._keep_last_n:
                for path in self._saved_pairs.popleft():
                    path.unlink(missing_ok=True)
                    if self.verbose:
                        print(f"[CHECKPOINT] Pruned owned artifact: {path.name}")
        return result


def write_training_result(log_dir, model, args_cli):
    from config.reward_units import reward_error_source
    from evaluation.provenance import checkpoint_identity, source_identity, runtime_assets
    checkpoint = Path(log_dir) / "model.zip"
    normalizer = Path(log_dir) / "vecnormalize.pkl"
    identity = checkpoint_identity(checkpoint)
    for field in ('training_manifest_verified', 'training_manifest_sha256', 'training_source_sha256'):
        identity.pop(field, None)  # These describe a consumer's link, not a self-signed manifest.
    result = {
        'schema_version': 2,
        **identity,
        'training_seed': int(args_cli.seed), 'training_seed_source': 'run_args',
        'source': source_identity(),
        'runtime_assets': runtime_assets(args_cli.yolo_model_path),
        "run_id": Path(log_dir).name,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
        "normalizer": str(normalizer), "normalizer_sha256": file_digest(normalizer),
        "seed": int(args_cli.seed), "num_envs": int(args_cli.num_envs),
        "steps_requested": int(args_cli.max_steps), "steps_actual": int(model.num_timesteps),
        "training_complete": int(model.num_timesteps) >= int(args_cli.max_steps),
        "resumed": bool(args_cli.checkpoint),
        "comparison_protocol": os.environ.get("ABLATION_PROTOCOL") or "standalone",
        "reward_error_source": reward_error_source(args_cli.asymmetric_critic),
        "decomposed_advantage": bool(args_cli.decomposed_advantage),
        "asymmetric_critic": bool(args_cli.asymmetric_critic),
        "value_head_outputs": int(model.policy.value_net.out_features),
    }
    policy_stats = model.policy.asymmetric_debug_stats() if hasattr(model.policy, 'asymmetric_debug_stats') else {}
    raw_width = int(model.observation_space.shape[0])
    result['policy_features'] = {
        'raw_observation_width': raw_width,
        'actor_observation_width': int(policy_stats.get('actor_obs_dim', raw_width)),
        'active_critic_features': int(policy_stats.get('critic_obs_dim', raw_width)),
        'privileged_observation_width': int(policy_stats.get('privileged_obs_dim', 0)),
    }
    atomic_json(Path(log_dir) / "training_result.json", result)
    handoff = os.environ.get("TRAIN_RUN_RESULT")
    if handoff:
        atomic_json(handoff, result)
    return result
