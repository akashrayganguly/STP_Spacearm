"""Safety geometry (DESIGN §6.4). Phase 5: analytic sphere-capsule distances used by the environment's
model-based obstacle features. The `SafetyShield` itself is added in Phase 7.
"""
from __future__ import annotations

import torch

_EPS = 1e-12


def sphere_capsule_distance(center, radius, a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Signed distance between a sphere (center (..., 3), radius) and capsules (segments a, b (..., K, 3),
    radii r (K,)): shape (..., K), negative = penetration. Differentiable (also at zero distance)."""
    c = torch.as_tensor(center, dtype=a.dtype, device=a.device)[..., None, :]
    ab = b - a
    s = (((c - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(_EPS)).clamp(0.0, 1.0)
    closest = a + s[..., None] * ab
    dist = torch.sqrt(((c - closest) ** 2).sum(-1) + _EPS)
    return dist - r - radius


def obstacle_clearance(kin, q, center, radius) -> torch.Tensor:
    """Minimum signed distance (...) between the arm capsules at joint angles q (..., 7) and a sphere
    obstacle (body-frame center (3,), radius), on the exact FK."""
    a, b, r = kin.capsules(q)
    return sphere_capsule_distance(center, radius, a, b, r).min(-1).values
