"""Cartesian-only control and checkpoint checks, safe before Isaac launches."""
from pathlib import Path
import json
import zipfile


def require_cartesian_mode(mode=None) -> str:
    value = str(mode or "cartesian").strip().lower()
    if value != "cartesian":
        raise ValueError(f"Only Cartesian XYZ control is supported; unsupported control mode: {mode!r}.")
    return "cartesian"


def validate_cartesian_metadata(cfg: dict) -> None:
    for key in ("control_mode", "mode"):
        if cfg.get(key) is not None:
            require_cartesian_mode(cfg[key])
    if cfg.get("action_dim") is not None and cfg["action_dim"] != 3:
        raise ValueError("A 3D Cartesian XYZ action space is required.")


def validate_cartesian_checkpoint(path, cfg=None) -> None:
    """Reject legacy joint checkpoints, including those without control metadata.

    SB3's JSON exposes the action shape without unpickling policy classes or
    importing Torch. Existing Cartesian checkpoints need no new metadata.
    """
    validate_cartesian_metadata(cfg or {})
    if not path:
        return
    checkpoint = Path(path)
    if not checkpoint.exists() and checkpoint.suffix != ".zip":
        checkpoint = Path(str(checkpoint) + ".zip")
    tag = checkpoint.parent.name.lower()
    if "joint_position" in tag or "jointposition" in tag or tag.endswith("_j"):
        raise ValueError("Joint-position checkpoints are unsupported; only Cartesian XYZ is supported.")
    with zipfile.ZipFile(checkpoint) as archive:
        data = json.loads(archive.read("data"))
    validate_cartesian_metadata(data)
    shape = (data.get("action_space") or {}).get("_shape")
    if shape != [3]:
        raise ValueError(
            f"Unsupported checkpoint action shape {shape!r}; Cartesian XYZ [3] is required."
        )
