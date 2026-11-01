"""Checkpoint-metadata inference for play (pure, no Isaac Sim).

These run BEFORE Isaac Sim launches: they recover
the actor/critic settings the checkpoint was trained with (from its
params/agent.yaml) so the env can be rebuilt to match the saved policy's obs/action
dims. No Isaac dependency; control is Cartesian-only. Edit here, not play.py.
"""

import glob
import os
import sys

from evaluation.agent_metadata import parse_agent_yaml
from rl_env.project_config import local_asset_path


def _resolve_checkpoint_from_argv() -> str | None:
    """Best-effort checkpoint path BEFORE argparse runs: scan argv for --checkpoint, else newest
    model.zip under the log base (project_config is config-free, safe to import early)."""
    for i, a in enumerate(sys.argv):
        if a == "--checkpoint" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--checkpoint="):
            return a.split("=", 1)[1]
    try:
        from rl_env.project_config import log_base

        cands = sorted(glob.glob(os.path.join(log_base(), "*/model.zip")))
        return cands[-1] if cands else None
    except Exception:
        return None


def restore_training_object_env(checkpoint_path: str | None = None) -> None:
    """Inject the checkpoint's training object into os.environ for manual replay.

    config/task.py + config/perception.py read those at IMPORT time (frozen), so this MUST be called
    BEFORE `config` (via config) is first imported — else a shape-trained policy would
    replay against the default cylinder. Detection filtering intentionally is not restored:
    every checkpoint uses the current canonical gripper mask, while an explicit process-level
    YOLO_ROI_ENABLED override remains available.
    """
    from evaluation.checkpoint_assets import restore_object_env
    restore_object_env(checkpoint_path or _resolve_checkpoint_from_argv())

def _load_checkpoint_agent_cfg_early(path: str | None) -> dict:
    if not path:
        print("[AGENT_CFG] No checkpoint path — using run_settings defaults for ppo_head/asymmetric_critic.", flush=True)
        return {}
    agent_path = os.path.join(os.path.dirname(path), "params", "agent.yaml")
    if not os.path.exists(agent_path):
        print(
            f"[AGENT_CFG][WARN] agent.yaml not found: {agent_path}\n"
            "  → ppo_head/asymmetric_critic will use run_settings defaults, NOT the checkpoint's actual settings.\n"
            "  → If the loaded policy crashes or has wrong obs_dim, this is likely the cause.",
            flush=True,
        )
        return {}
    try:
        with open(agent_path, "r") as f:
            text = f.read()
        data = parse_agent_yaml(text, on_legacy=lambda error: print(
            f"[AGENT_CFG][WARN] safe_load failed, retrying unsafe_load: {error}", flush=True))
        if not isinstance(data, dict):
            print(f"[AGENT_CFG][WARN] agent.yaml is not a dict: {agent_path}", flush=True)
            return {}
        print(
            f"[AGENT_CFG] Loaded from {agent_path}:\n"
            f"  ppo_head={data.get('ppo_head', '<missing>')}  "
            f"asymmetric_critic={data.get('asymmetric_critic', '<missing>')}  "
            f"actor_obs_dim={data.get('actor_obs_dim', '<missing>')}  "
            f"critic_obs_dim={data.get('critic_obs_dim', '<missing>')}  "
            f"privileged_obs_dim={data.get('privileged_obs_dim', '<missing>')}",
            flush=True,
        )
        return data
    except Exception as exc:
        print(f"[AGENT_CFG][WARN] Failed to read {agent_path}: {exc}", flush=True)
        return {}


def _infer_ppo_head_from_agent_cfg(agent_cfg: dict) -> str:
    from config.policy import validate_supported_architecture
    validate_supported_architecture(agent_cfg)
    return "single"
