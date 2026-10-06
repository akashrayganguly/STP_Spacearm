"""Outcome losses for Level-1 trajectory learning (DESIGN §5.3, D7): no labelled answers needed.

loss = w_reach * |FK(q_goal) - target|^2 + w_coll * mean(relu(margin - clearance)^2) + w_smooth * sum |dQ|^2
"""
from __future__ import annotations

import torch


def reach_loss(tcp_goal: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared TCP error at the goal (m^2)."""
    return ((tcp_goal - target) ** 2).sum(-1).mean()


def collision_loss(clearance: torch.Tensor, margin: float) -> torch.Tensor:
    """Mean squared shortfall of the clearance below `margin` (zero when every point keeps the margin)."""
    return torch.relu(margin - clearance).pow(2).mean()


def smoothness_loss(Q: torch.Tensor) -> torch.Tensor:
    """Sum over the path of squared joint steps, averaged over the batch. Q: (..., T, 7)."""
    dQ = Q[..., 1:, :] - Q[..., :-1, :]
    return dQ.pow(2).sum((-1, -2)).mean()


def limit_violation(Q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Total amount (rad) by which the path leaves the joint limits (0 when it stays inside)."""
    return (torch.relu(Q - upper) + torch.relu(lower - Q)).sum()


def level1_loss(Q: torch.Tensor, tcp_goal: torch.Tensor, target: torch.Tensor, clearance: torch.Tensor,
                cfg: dict) -> tuple[torch.Tensor, dict]:
    """(total, parts): Q (B, T, 7) sampled path, tcp_goal/target (B, 3), clearance (B, T) from the DistanceNet."""
    c = cfg["traj"]
    parts = {"reach": reach_loss(tcp_goal, target),
             "coll": collision_loss(clearance, cfg["collision"]["plan_margin"]),
             "smooth": smoothness_loss(Q)}
    total = c["w_reach"] * parts["reach"] + c["w_coll"] * parts["coll"] + c["w_smooth"] * parts["smooth"]
    return total, {k: v.detach() for k, v in parts.items()}
