"""Require a checkpoint trained with the paper preset before a matched evaluation."""
import json
import os
import math
from pathlib import Path
import zipfile

from config.policy import validate_supported_architecture
from config.reward_units import reward_error_source, reward_units_of, scalar_reward_units_of
from config.experiment import PROTOCOL
from evaluation.provenance import METHODS


def require_checkpoint_method(checkpoint, metadata, method=None):
    """Historical replay remains available without a matched-method request."""
    if method is None and os.environ.get("ABLATION_PROTOCOL") == PROTOCOL:
        arm = os.environ.get("ABLATION10_ARM", "")
        method = next((name for name, row in METHODS.items() if str(row[0]) == arm), None)
        if method is None:
            raise ValueError("Matched evaluation requires a known ABLATION10_ARM or explicit --method")
    if method is None:
        if metadata.get("comparison_protocol") != PROTOCOL:
            return None
        method = next((name for name, row in METHODS.items() if row[0] == metadata.get("ablation10_arm")), None)
        if method is None:
            raise ValueError("Matched checkpoint has an unknown baseline arm")
    arm, asymmetric, decomposed = METHODS[method]
    validate_supported_architecture(metadata)
    expected = {"comparison_protocol": PROTOCOL, "ablation10_arm": arm,
                "asymmetric_critic": asymmetric, "decomposed_advantage": decomposed,
                "value_head_outputs": 1, "reward_stats_units": "per_step",
                "reward_error_source": "gt"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Wrong {method} checkpoint: {key}={metadata.get(key)!r}, expected {value!r}")
    if reward_error_source(asymmetric) != expected["reward_error_source"]:
        raise ValueError("Explicit reward source does not match the assigned baseline")
    with zipfile.ZipFile(checkpoint) as archive:
        data = json.loads(archive.read("data"))
    kwargs = data.get("policy_kwargs", {})
    credit = kwargs.get("decomposed_config", {})
    if bool(credit.get("decomposed_advantage", False)) != decomposed:
        raise ValueError("Saved model credit mode disagrees with checkpoint metadata")
    if kwargs.get("actor_obs_dim") != (12 if asymmetric else None):
        raise ValueError("Saved actor mask disagrees with the assigned baseline")
    for key, shape in (("action_space", [3]), ("observation_space", [18 if asymmetric else 12])):
        if (data.get(key) or {}).get("_shape") != shape:
            raise ValueError(f"Wrong {method} {key} shape")
    if decomposed:
        if reward_units_of(credit) != "legacy_rate" or not credit.get("scalar_critic", False):
            raise ValueError("Matched credit requires legacy units and a scalar value head")
        if tuple(credit.get("decomposed_axis_weights", (1, 1, 1))) != (1, 1, 1):
            raise ValueError("Matched baseline requires unit axis weights")
    elif scalar_reward_units_of(data) != "legacy_rate" or metadata.get("scalar_reward_units") != "legacy_rate":
        raise ValueError("Matched scalar PPO requires legacy learner units")
    return method


def require_normalizer_method(normalizer, model, method, step_dt=None):
    if method is None:
        return
    decomposed = METHODS[method][2]
    from stable_baselines3.common.vec_env import VecNormalize
    from training.reward_normalization import PpoRewardVecNormalize
    if decomposed:
        if type(normalizer) is not VecNormalize:
            raise ValueError("Matched per-axis credit requires ordinary VecNormalize")
    else:
        if not isinstance(normalizer, PpoRewardVecNormalize):
            raise ValueError("Matched scalar PPO requires PpoRewardVecNormalize")
        saved_dt = float(normalizer.reward_step_dt)
        if not math.isfinite(saved_dt) or saved_dt <= 0:
            raise ValueError("Saved scalar reward step_dt is invalid")
        if step_dt is not None and not math.isclose(saved_dt, float(step_dt), rel_tol=1e-9):
            raise ValueError("Scalar reward step_dt differs from the current environment")
    actual = getattr(normalizer, "scalar_reward_units", "per_step")
    if actual != ("per_step" if decomposed else "legacy_rate"):
        raise ValueError("Normalizer class/units disagree with the assigned baseline")
    if model.policy.value_net.out_features != 1:
        raise ValueError("Matched baseline requires one actual value output")
