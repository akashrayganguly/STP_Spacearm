"""DistanceNet (DESIGN §5.2, D6): joint angles -> signed (d_body, d_self) and a conservative clearance.

* Input features [sin q, cos q]: FK is built from sin/cos, so these are natural and 2*pi-periodic.
* SiLU MLP: smooth gradients d clearance / d q for the planner, the prior's null-space push and the shield.
* clearance = -tau * logsumexp(-d / tau) over the two heads: a smooth minimum that never exceeds the
  smaller head (and is at most tau*log(2) below it).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceNet(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 3, n_joints: int = 7, tau: float = 0.01):
        super().__init__()
        self.n_joints, self.tau = n_joints, float(tau)
        dims = [2 * n_joints] + [hidden] * layers
        mods: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.SiLU()]
        mods.append(nn.Linear(dims[-1], 2))
        self.mlp = nn.Sequential(*mods)

    @classmethod
    def from_config(cls, cfg: dict) -> "DistanceNet":
        c = cfg["distance_net"]
        return cls(hidden=c["hidden"], layers=c["layers"], n_joints=len(cfg["robot"]["arm"]["axes"]), tau=c["tau"])

    @staticmethod
    def features(q: torch.Tensor) -> torch.Tensor:
        """(..., 7) joint angles -> (..., 14) = [sin q, cos q]."""
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """(..., 7) -> (..., 2) predicted signed distances (d_body, d_self) in metres.

        Inputs of another float dtype (e.g. float64 from exact kinematics) are cast to the weights' dtype
        and the output is cast back, so callers do not have to care (the cast is differentiable)."""
        w = self.mlp[0].weight
        out = self.mlp(self.features(q.to(dtype=w.dtype, device=w.device)))
        return out.to(dtype=q.dtype) if q.is_floating_point() else out

    def clearance(self, q: torch.Tensor) -> torch.Tensor:
        """(..., 7) -> (...) smooth minimum of the two heads (conservative: <= min(d_body, d_self))."""
        return -self.tau * torch.logsumexp(-self(q) / self.tau, dim=-1)


def distance_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Huber loss (beta = huber_beta) on clipped targets, x near_weight where the true |d| < near_band.

    A plain weighted mean (not normalised by the weights), so an error near the collision surface
    always costs more than the same error far away."""
    c = cfg["distance_net"]
    lo, hi = c["clip"]
    target = target.clamp(lo, hi)
    w = torch.where(target.abs() < c["near_band"], torch.full_like(target, c["near_weight"]), torch.ones_like(target))
    return (w * F.smooth_l1_loss(pred, target, reduction="none", beta=c["huber_beta"])).mean()
