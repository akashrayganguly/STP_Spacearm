"""Classical reflex prior (DESIGN §6.2, D10): the layer the RL residual sits on.

qdot = J+ v  +  (I - J+ J) k_null w_s grad(d_self)  +  k_obs w_o grad(d_obs)
  v   = clip_norm(k_p * (TCP -> target), v_max)              Cartesian reaching (damped least squares)
  w_s = ramp 0 -> 1 as the self-clearance falls below null_activation   (null space: the hand does not move)
  w_o = ramp 0 -> 1 as the obstacle clearance falls below obs_activation (repulsion reflex)
Everything uses only onboard measurements (biased encoders, vision), so it can run on the robot.
"""
from __future__ import annotations

import numpy as np
import torch


def _ramp(clearance: float, activation: float) -> float:
    return float(np.clip((activation - clearance) / activation, 0.0, 1.0))


def reach_prior(kin, q_meas, rel_target_body, cfg: dict, clearance=None, clearance_grad=None,
                obstacle_clearance=None, obstacle_grad=None, J=None) -> np.ndarray:
    """Normalised joint-velocity command in [-1, 1]^7 (x max_velocity = rad/s).

    rel_target_body: TCP -> target vector in the body frame (m). clearance / obstacle_clearance in m,
    gradients = joint-space directions that increase the clearance (unit length expected).
    J: optional precomputed TCP Jacobian (3, 7) at q_meas (saves an FK pass inside the environment).
    The command is scaled down as a whole if any joint would exceed the speed limit (keeps the direction)."""
    c = cfg["prior"]
    v_joint = cfg["robot"]["arm"]["max_velocity"]
    q = np.asarray(q_meas, float)
    if J is None:
        with torch.no_grad():
            J = kin.jacobian(torch.as_tensor(q, dtype=kin.dtype)).double().numpy()    # (3, 7)
    v = c["k_p"] * np.asarray(rel_target_body, float)
    n = np.linalg.norm(v)
    if n > c["v_max"]:
        v *= c["v_max"] / n
    J_pinv = J.T @ np.linalg.inv(J @ J.T + c["damping"] ** 2 * np.eye(3))              # (7, 3)
    qd = J_pinv @ v
    if clearance is not None and clearance_grad is not None:
        w_s = _ramp(float(clearance), c["null_activation"])
        if w_s > 0:
            qd += (np.eye(len(q)) - J_pinv @ J) @ (c["k_null"] * w_s * np.asarray(clearance_grad, float))
    if obstacle_clearance is not None and obstacle_grad is not None:
        w_o = _ramp(float(obstacle_clearance), c["obs_activation"])
        if w_o > 0:
            qd += c["k_obs"] * w_o * np.asarray(obstacle_grad, float)
    cmd = qd / v_joint
    peak = np.abs(cmd).max()
    return cmd / peak if peak > 1.0 else cmd
