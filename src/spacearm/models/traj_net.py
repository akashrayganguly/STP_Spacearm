"""TrajNet (DESIGN §5.3, D7/D8): (q_start, target) -> Bezier control points of a joint-space path.

* Degree-7 Bezier with P0 = P1 = q_start and P6 = P7 = goal: rest-to-rest motion (zero velocity at both ends).
* Interior points and the goal live in an unconstrained "u" space and are squashed into the joint limits
  (squash = mid + half * tanh). Every control point is inside the limits, so the whole curve is too
  (convex-hull property of Bezier curves): joint limits hold by construction.
* Output = pre-squash goal u_goal (7) + offsets delta (4 x 7) from the straight line in u space.
  The last layer starts at zero, so an untrained net proposes "go to the middle of the ranges".
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import torch
import torch.nn as nn

_UNSQUASH_EPS = 1e-6


@lru_cache(maxsize=64)
def _basis_cached(T: int, degree: int, dtype: torch.dtype) -> torch.Tensor:
    s = torch.linspace(0.0, 1.0, T, dtype=torch.float64)[:, None]
    k = torch.arange(degree + 1, dtype=torch.float64)[None, :]
    coef = torch.tensor([comb(degree, i) for i in range(degree + 1)], dtype=torch.float64)[None, :]
    return (coef * s.pow(k) * (1.0 - s).pow(degree - k)).to(dtype)


def bezier_basis(T: int, degree: int = 7, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Bernstein basis (T, degree + 1) at T evenly spaced curve parameters s in [0, 1]."""
    return _basis_cached(int(T), int(degree), dtype)


def squash(u: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Unconstrained u -> joint angles strictly inside [lower, upper]."""
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    return torch.clamp(mid + half * torch.tanh(u), lower, upper)


def unsquash(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Inverse of `squash` (joint angles at a limit map to a large but finite u)."""
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    return torch.atanh(((q - mid) / half).clamp(-1.0 + _UNSQUASH_EPS, 1.0 - _UNSQUASH_EPS))


def make_control_points(q_start: torch.Tensor, u_goal: torch.Tensor, delta: torch.Tensor,
                        lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Control points (..., degree + 1, 7) from q_start (..., 7), u_goal (..., 7), delta (..., degree - 3, 7).

    P0 = P1 = q_start, P_k = squash(lerp(unsquash(q_start), u_goal, k / degree) + delta_k) for the interior,
    P_{n-1} = P_n = squash(u_goal)."""
    n_int = delta.shape[-2]
    degree = n_int + 3
    u_start = unsquash(q_start, lower, upper)
    s = torch.arange(2, degree - 1, dtype=q_start.dtype, device=q_start.device)[:, None] / degree   # (n_int, 1)
    interior = squash(u_start[..., None, :] + s * (u_goal - u_start)[..., None, :] + delta, lower, upper)
    goal = squash(u_goal, lower, upper)[..., None, :]
    start = q_start[..., None, :]
    return torch.cat([start, start, interior, goal, goal], dim=-2)


def curve(P: torch.Tensor, T: int) -> torch.Tensor:
    """Sample the Bezier curve with control points P (..., n + 1, 7) at T points -> (..., T, 7)."""
    B = bezier_basis(T, P.shape[-2] - 1, P.dtype).to(P.device)
    return torch.einsum("tk,...kj->...tj", B, P)


class TrajNet(nn.Module):
    """[sin q_start, cos q_start, target] (17) -> hidden x layers SiLU -> u_goal (7) + delta ((degree - 3) x 7)."""

    def __init__(self, lower, upper, hidden: int = 512, layers: int = 3, degree: int = 7):
        super().__init__()
        lower, upper = torch.as_tensor(lower, dtype=torch.float32), torch.as_tensor(upper, dtype=torch.float32)
        self.register_buffer("lower", lower.clone())          # buffers: not trainable, saved in the state_dict
        self.register_buffer("upper", upper.clone())
        self.n_joints, self.degree = len(lower), int(degree)
        self.n_interior = self.degree - 3
        dims = [2 * self.n_joints + 3] + [hidden] * layers
        mods: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.SiLU()]
        last = nn.Linear(dims[-1], self.n_joints * (1 + self.n_interior))
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        mods.append(last)
        self.mlp = nn.Sequential(*mods)

    @classmethod
    def from_config(cls, cfg: dict, lower, upper) -> "TrajNet":
        c = cfg["traj"]
        return cls(lower, upper, hidden=c["hidden"], layers=c["layers"], degree=c["degree"])

    def raw(self, q_start: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-squash outputs: u_goal (..., 7) and delta (..., degree - 3, 7)."""
        x = torch.cat([torch.sin(q_start), torch.cos(q_start), target], dim=-1)
        out = self.mlp(x)
        return out[..., : self.n_joints], out[..., self.n_joints:].reshape(*out.shape[:-1], self.n_interior,
                                                                          self.n_joints)

    def forward(self, q_start: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Control points (..., degree + 1, 7), all inside the joint limits."""
        u_goal, delta = self.raw(q_start, target)
        lo, hi = self.lower.to(q_start.dtype), self.upper.to(q_start.dtype)
        return make_control_points(q_start, u_goal, delta, lo, hi)
