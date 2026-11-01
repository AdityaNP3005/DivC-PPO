from __future__ import annotations

import torch


def per_axis_in_band(
    error_xyz: torch.Tensor,
    xy_tol_m: float,
    z_tol_m: float,
) -> torch.Tensor:
    """Return whether each XYZ error is inside the inclusive per-axis band."""
    error = error_xyz.abs()
    return (
        (error[..., 0] <= float(xy_tol_m))
        & (error[..., 1] <= float(xy_tol_m))
        & (error[..., 2] <= float(z_tol_m))
    )


def advance_stable_success(
    streak: torch.Tensor,
    stable: torch.Tensor,
    at_target: torch.Tensor,
    just_reset: torch.Tensor,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance consecutive in-band streak and sticky per-episode success."""
    at_target = at_target.to(torch.bool)
    just_reset = just_reset.to(torch.bool)
    current = torch.where(at_target, streak + 1, torch.zeros_like(streak))
    current = torch.where(just_reset, torch.zeros_like(streak), current)
    reached = current >= max(int(required_steps), 1)
    stable = torch.where(just_reset, reached, stable | reached)
    return current, stable
