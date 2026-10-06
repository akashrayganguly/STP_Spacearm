"""Safety geometry and the model-based safety shield (DESIGN §6.4, D15).

The shield sits after the policy: it predicts where a command takes the arm in the next
`lookahead_steps` control steps, q' = q + s * a * v_max * lookahead / control_hz, for s in `scales`
(1, 0.5, 0.25), and accepts the largest s with h(q') >= 0, where

    h(q) = min(self clearance(q) - self_margin, obstacle clearance(q) - obstacle_margin)

(self clearance = DistanceNet on the measured angles; obstacle clearance = analytic sphere-capsule
distance on the exact FK). If no scale is safe it commands an escape, escape_speed * grad h / |grad h|_inf,
or zero if escaping does not increase h. Honest limit: it only knows the *measured* joint angles and its
models, so it is a filter, not a proof.
"""
from __future__ import annotations

import numpy as np
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


class SafetyShield:
    """Filter normalised joint-velocity commands so that the predicted configuration keeps the margins."""

    def __init__(self, kin, cfg: dict, self_clearance_fn=None):
        s = cfg["shield"]
        self.kin, self.self_clearance_fn = kin, self_clearance_fn
        self.self_margin, self.obstacle_margin = float(s["self_margin"]), float(s["obstacle_margin"])
        self.scales = [float(x) for x in s["scales"]]
        self.escape_speed = float(s["escape_speed"])
        self.step = float(cfg["robot"]["arm"]["max_velocity"]) * s["lookahead_steps"] / cfg["sim"]["control_hz"]

    def safety_value(self, q: torch.Tensor, obstacle=None) -> torch.Tensor:
        """h(q) (...): >= 0 means both margins are kept; +inf if there is nothing to check."""
        q = torch.as_tensor(q, dtype=self.kin.dtype)
        h = torch.full(q.shape[:-1], float("inf"), dtype=self.kin.dtype)
        if self.self_clearance_fn is not None:
            h = torch.minimum(h, self.self_clearance_fn(q).to(self.kin.dtype) - self.self_margin)
        if obstacle is not None:
            c = obstacle_clearance(self.kin, q, torch.as_tensor(np.asarray(obstacle["center"], float)), obstacle["radius"])
            h = torch.minimum(h, c - self.obstacle_margin)
        return h

    def filter(self, q_meas, action, obstacle=None):
        """(safe action (7,), info). info: intervened, scale (accepted s, 0 for escape/stop), escape, h_pred."""
        q = torch.as_tensor(np.asarray(q_meas, float), dtype=self.kin.dtype)
        a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        if self.self_clearance_fn is None and obstacle is None:
            return a, {"intervened": False, "scale": 1.0, "escape": False, "h_pred": float("inf")}
        at = torch.as_tensor(a, dtype=self.kin.dtype)
        scales = torch.tensor(self.scales, dtype=self.kin.dtype)
        with torch.no_grad():
            h = self.safety_value(q + scales[:, None] * at * self.step, obstacle)      # all candidates at once
        for k, s in enumerate(self.scales):
            if h[k] >= 0:
                return a * s, {"intervened": s < 1.0, "scale": s, "escape": False, "h_pred": float(h[k])}

        # No scale is safe: move along the gradient of h (direction that increases the clearance).
        qg = q.clone().requires_grad_(True)
        h0 = self.safety_value(qg, obstacle)
        (grad,) = torch.autograd.grad(h0, qg, allow_unused=True)        # leaves the DistanceNet's .grad alone
        h0 = h0.detach()
        g = grad.double().numpy() if grad is not None else np.zeros(len(a))
        peak = np.abs(g).max()
        if not np.isfinite(peak) or peak < _EPS:
            return np.zeros_like(a), {"intervened": True, "scale": 0.0, "escape": False, "h_pred": float(h0)}
        esc = self.escape_speed * g / peak
        with torch.no_grad():
            h_esc = self.safety_value(q + torch.as_tensor(esc, dtype=self.kin.dtype) * self.step, obstacle)
        if h_esc > h0:
            return esc, {"intervened": True, "scale": 0.0, "escape": True, "h_pred": float(h_esc)}
        return np.zeros_like(a), {"intervened": True, "scale": 0.0, "escape": False, "h_pred": float(h0)}


def shield_from_config(kin, cfg: dict, dnet) -> SafetyShield:
    """Shield whose self clearance is the trained DistanceNet (conservative smooth min of both heads)."""
    dnet.eval()

    def self_clearance(q: torch.Tensor) -> torch.Tensor:
        return dnet.clearance(q)

    return SafetyShield(kin, cfg, self_clearance_fn=self_clearance)
