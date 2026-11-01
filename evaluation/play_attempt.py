"""Provenance and completion for a rendered demo, separate from validation metrics."""
from pathlib import Path

from evaluation.validation_contract import (
    atomic_json, create_evaluation_directory, evaluation_context, require_unchanged_assets,
)


def start_play_attempt(checkpoint, output, seed, episodes, steps_per_episode, video_enabled):
    if episodes <= 0 or steps_per_episode <= 0:
        raise ValueError("Play episode count and step budget must be positive")
    context = evaluation_context(checkpoint, seed, 1)
    output = create_evaluation_directory(checkpoint, output)
    status = {
        **context, "attempt_kind": "play", "play_id": Path(output).name,
        "play_complete": False, "reason": "pending", "exit_code": None,
        "episodes_requested": episodes, "episodes_completed": 0,
        "steps_per_episode": steps_per_episode, "video_enabled": bool(video_enabled),
    }
    atomic_json(Path(output) / "play_status.json", status)
    return status


def finish_play_attempt(output, status, completed, lengths, frame_counts, videos):
    require_unchanged_assets(status)
    complete = completed >= status["episodes_requested"]
    result = {
        **status, "play_complete": complete,
        "reason": "episode_budget_met" if complete else "episode_budget_not_met",
        "exit_code": 0 if complete else 2,
        "episodes_completed": completed, "episode_steps": list(lengths),
        "steps_run": sum(lengths), "captured_frames": dict(frame_counts),
        # Only encoders that returned successfully publish a video here. Capture
        # counts are not a promise of decoder validity or successful detection.
        "encoded_videos": list(videos),
    }
    atomic_json(Path(output) / "play_status.json", result)
    return result
