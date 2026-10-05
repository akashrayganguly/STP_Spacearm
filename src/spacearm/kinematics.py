"""Exact, batched, differentiable forward kinematics of the arm in the body frame (PyTorch).

Joint frame chain (DESIGN §4 / lesson 7):
    t <- t + R[:, z] * offset_i        (offset_0 = pedestal length, offset_i = link_lengths[i-1])
    R <- R @ Rot(axis_i, q_i)
    TCP = t_7 + R_7[:, z] * tool_offset
The frames coincide with PyBullet's URDF link frames when the base is at identity.
"""
from __future__ import annotations

import numpy as np
import torch

from spacearm.robot_model import joint_limits

_AXIS_ID = {"x": 0, "y": 1, "z": 2}


def _rotation(axis: int, q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) about a principal axis by angles q (...)."""
    c, s = torch.cos(q), torch.sin(q)
    one, zero = torch.ones_like(q), torch.zeros_like(q)
    if axis == 0:
        rows = [one, zero, zero, zero, c, -s, zero, s, c]
    elif axis == 1:
        rows = [c, zero, s, zero, one, zero, -s, zero, c]
    else:
        rows = [c, -s, zero, s, c, zero, zero, zero, one]
    return torch.stack(rows, dim=-1).reshape(*q.shape, 3, 3)


class ArmKinematics:
    """FK, analytic Jacobian and capsule segments for joint angles of shape (..., 7)."""

    def __init__(self, cfg: dict, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
        r, arm = cfg["robot"], cfg["robot"]["arm"]
        self.dtype, self.device = dtype, torch.device(device)
        kw = {"dtype": dtype, "device": self.device}
        self.axes = [_AXIS_ID[a] for a in arm["axes"]]
        self.n_joints = len(self.axes)
        self.mount = torch.tensor(r["mount_xyz"], **kw)
        self.offsets = [float(r["pedestal"]["length"])] + [float(x) for x in arm["link_lengths"][:-1]]
        self.link_lengths = torch.tensor(arm["link_lengths"], **kw)
        self.radii = torch.tensor(arm["link_radii"], **kw)
        self.tool_offset = float(arm["tool_offset"])
        lo, hi = joint_limits(cfg)
        self.lower, self.upper = torch.tensor(lo, **kw), torch.tensor(hi, **kw)

    def _as_q(self, q) -> torch.Tensor:
        if isinstance(q, torch.Tensor):
            return q.to(dtype=self.dtype, device=self.device)      # differentiable cast
        return torch.as_tensor(np.asarray(q), dtype=self.dtype, device=self.device)

    def forward(self, q) -> tuple[torch.Tensor, torch.Tensor]:
        """Link frames link1..link7 + TCP: R (..., 8, 3, 3) body-frame rotations, t (..., 8, 3) origins."""
        q = self._as_q(q)
        batch = q.shape[:-1]
        R = torch.eye(3, dtype=self.dtype, device=self.device).expand(*batch, 3, 3)
        t = self.mount.expand(*batch, 3)
        Rs, ts = [], []
        for i, axis in enumerate(self.axes):
            t = t + R[..., :, 2] * self.offsets[i]
            R = R @ _rotation(axis, q[..., i])
            Rs.append(R)
            ts.append(t)
        Rs.append(R)
        ts.append(t + R[..., :, 2] * self.tool_offset)
        return torch.stack(Rs, dim=-3), torch.stack(ts, dim=-2)

    def tcp(self, q) -> torch.Tensor:
        """TCP position (..., 3) in the body frame."""
        return self.forward(q)[1][..., -1, :]

    def jacobian(self, q) -> torch.Tensor:
        """Analytic positional Jacobian d tcp / d q, shape (..., 3, 7): column i = z_i x (p_tcp - p_i)."""
        R, t = self.forward(q)
        z = torch.stack([R[..., i, :, a] for i, a in enumerate(self.axes)], dim=-2)     # (..., 7, 3)
        r = t[..., -1:, :] - t[..., : self.n_joints, :]                                  # (..., 7, 3)
        return torch.linalg.cross(z, r, dim=-1).transpose(-1, -2)

    def capsules(self, q) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Arm capsules: segment start a (..., 7, 3), end b (..., 7, 3), radii r (7,)."""
        R, t = self.forward(q)
        a = t[..., : self.n_joints, :]
        b = a + R[..., : self.n_joints, :, 2] * self.link_lengths[:, None]
        return a, b, self.radii


def tcp_numpy(kin: ArmKinematics, q) -> np.ndarray:
    """NumPy convenience wrapper: joint angles (..., 7) -> TCP position (..., 3)."""
    with torch.no_grad():
        return kin.tcp(q).cpu().numpy()
