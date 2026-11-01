from __future__ import annotations

"""Single source for single-actor Cartesian observation widths."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ObsLayout:
    asymmetric_critic: bool
    depth_in_obs: bool
    yolo_obs_dim: int
    actor_base_dim: int

def resolve_obs_layout(asymmetric_critic: bool, actor_depth_enabled: bool = True) -> ObsLayout:
    depth = not asymmetric_critic or actor_depth_enabled
    return ObsLayout(bool(asymmetric_critic), bool(depth), 3 if depth else 2, 12 if depth else 10)
