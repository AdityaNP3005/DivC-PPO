"""Evaluation/play package.

Pure helpers split out of the play.py script (host-importable, no Isaac Sim):

    checkpoint_rebuild — recover actor/critic settings from a
                         checkpoint's agent.yaml before Isaac launches
    camera_capture     — RGB normalization + drawing the cached YOLO detection

The live camera capture, env build, eval loop, and video/summary writing remain
in the top-level play.py script (they thread module-global Isaac runtime state).
"""

from evaluation.checkpoint_rebuild import (
    _infer_ppo_head_from_agent_cfg,
    _load_checkpoint_agent_cfg_early,
    restore_training_object_env,
)

__all__ = [
    "_load_checkpoint_agent_cfg_early",
    "_infer_ppo_head_from_agent_cfg",
    "restore_training_object_env",
]
