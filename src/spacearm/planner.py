"""Level-1 planner (DESIGN §5.4, D9): learned proposal -> exact refinement -> verification.

plan(q_start, target): TrajNet proposal -> refine (Adam on the same outcome loss, on the Bezier
parameters) -> polish the goal with damped-least-squares IK on the exact FK -> sample t_verify points.
plan_and_verify: exact PyBullet check; if it fails, plan once more with more refinement steps.
All planning happens in the spacecraft body frame (DESIGN D2).
"""
from __future__ import annotations

import time

import numpy as np
import pybullet as p
import torch

from spacearm.kinematics import ArmKinematics
from spacearm.losses import level1_loss
from spacearm.models.traj_net import curve, make_control_points, squash, unsquash

_LIMIT_TOL = 1e-9


# ------------------------------------------------------------------ queries
def free_pool(data: dict, margin: float) -> np.ndarray:
    """Configurations from a labelled dataset whose true clearance exceeds `margin` (N, 7), float64."""
    d_min = np.minimum(data["d_body"], data["d_self"])
    return np.asarray(data["q"][d_min > margin], dtype=np.float64)


def sample_queries(pool: np.ndarray, kin: ArmKinematics, n: int, rng: np.random.Generator, min_dist: float):
    """n planning queries (q_start (n,7), target (n,3), q_goal (n,7)): start and goal from the pool,
    target = FK(goal) (always reachable), start TCP at least `min_dist` from the target."""
    qs, tg, qg = [], [], []
    while len(qs) < n:
        i, j = rng.integers(0, len(pool), 2)
        if i == j:
            continue
        with torch.no_grad():
            t_s, t_g = kin.tcp(torch.tensor(pool[[i, j]], dtype=kin.dtype)).double().numpy()
        if np.linalg.norm(t_g - t_s) < min_dist:
            continue
        qs.append(pool[i])
        tg.append(t_g)
        qg.append(pool[j])
    return np.array(qs), np.array(tg), np.array(qg)


# ------------------------------------------------------------------ IK polish
def polish_ik(kin: ArmKinematics, q: torch.Tensor, target: torch.Tensor, iters: int = 10,
              damping: float = 0.01) -> torch.Tensor:
    """Damped-least-squares IK on the exact FK, clamped to the joint limits. q (..., 7), target (..., 3).
    (Defaults mirror traj.polish_iters / traj.polish_damping; the planner passes the config values.)"""
    q = torch.as_tensor(q, dtype=kin.dtype).clone()
    target = torch.as_tensor(target, dtype=kin.dtype)
    eye = torch.eye(3, dtype=kin.dtype)
    with torch.no_grad():
        for _ in range(int(iters)):
            J = kin.jacobian(q)                                             # (..., 3, 7)
            e = target - kin.tcp(q)                                         # (..., 3)
            A = J @ J.transpose(-1, -2) + damping ** 2 * eye
            dq = (J.transpose(-1, -2) @ torch.linalg.solve(A, e[..., None]))[..., 0]
            q = torch.clamp(q + dq, kin.lower, kin.upper)
    return q


# ------------------------------------------------------------------ planner
class Level1Planner:
    """Kinematics-aware planner: TrajNet proposal + refinement on the DistanceNet + IK polish."""

    def __init__(self, kin: ArmKinematics, dnet, tnet, cfg: dict):
        self.kin, self.dnet, self.tnet, self.cfg = kin, dnet, tnet, cfg
        self.c = cfg["traj"]
        self.dnet.eval()
        self.tnet.eval()

    def _points(self, qs, z_goal, z_delta):
        return make_control_points(qs, z_goal, z_delta, self.kin.lower, self.kin.upper)

    def plan(self, q_start, target, refine_steps: int | None = None, polish: bool = True):
        """Joint path (t_verify, 7) float64 from q_start to a goal whose TCP reaches `target` (body frame),
        and an info dict. refine_steps=0 and polish=False give the raw network proposal."""
        t0 = time.perf_counter()
        c, kin = self.c, self.kin
        steps = c["refine_steps"] if refine_steps is None else int(refine_steps)
        qs = torch.as_tensor(np.asarray(q_start, float), dtype=kin.dtype)[None]
        tg = torch.as_tensor(np.asarray(target, float), dtype=kin.dtype)[None]
        with torch.no_grad():
            u_goal, delta = self.tnet.raw(qs.to(torch.float32), tg.to(torch.float32))
        z_goal = u_goal.to(kin.dtype).clone().requires_grad_(True)
        z_delta = delta.to(kin.dtype).clone().requires_grad_(True)

        # Refine: a few Adam steps on the outcome loss, w.r.t. the Bezier parameters only
        # (autograd.grad keeps the networks' own .grad untouched).
        loss_val = float("nan")
        if steps > 0:
            opt = torch.optim.Adam([z_goal, z_delta], lr=c["refine_lr"])
            for _ in range(steps):
                P = self._points(qs, z_goal, z_delta)
                Q = curve(P, c["t_train"])
                loss, _ = level1_loss(Q, kin.tcp(P[:, -1]), tg, self.dnet.clearance(Q), self.cfg)
                z_goal.grad, z_delta.grad = torch.autograd.grad(loss, [z_goal, z_delta])
                opt.step()
                loss_val = loss.item()

        with torch.no_grad():
            q_goal = squash(z_goal, kin.lower, kin.upper)
            if polish:
                q_goal = polish_ik(kin, q_goal, tg, c["polish_iters"], c["polish_damping"])
            P = self._points(qs, unsquash(q_goal, kin.lower, kin.upper), z_delta)
            P[:, -2:] = q_goal[:, None, :]                   # exact polished goal (unsquash/squash round trip)
            Q = curve(P, c["t_verify"])[0]
            clear = self.dnet.clearance(Q)
            reach = torch.linalg.norm(kin.tcp(q_goal[0]) - tg[0]).item()
        Q = Q.double().numpy()
        Q[0] = np.asarray(q_start, float)                    # exact start (no float32 round-off)
        info = {"plan_time": time.perf_counter() - t0, "refine_steps": steps, "polish": polish,
                "pred_min_clearance": clear.min().item(), "fk_reach_err": reach, "refine_loss": loss_val,
                "control_points": P[0].double().numpy()}
        return Q, info


# ------------------------------------------------------------------ verification
def verify_trajectory(sim, Q, target, tol: float) -> dict:
    """Exact PyBullet check of a body-frame joint path: clearance > 0 at every point, final TCP within
    `tol` of the target, every point within the joint limits."""
    Q = np.asarray(Q, float)
    d = np.empty(len(Q))
    for k, q in enumerate(Q):
        sim.set_q(q)
        d[k] = min(sim.min_distances())
    reach = float(np.linalg.norm(sim.tcp_body() - np.asarray(target, float)))   # sim is at Q[-1]
    within = bool(np.all(Q >= sim.lower - _LIMIT_TOL) and np.all(Q <= sim.upper + _LIMIT_TOL))
    res = {"collision_free": bool(np.all(d > 0)), "min_dist": float(d.min()), "n_colliding": int(np.sum(d <= 0)),
           "reach_err": reach, "within_limits": within}
    res["success"] = res["collision_free"] and reach < tol and within
    return res


def plan_and_verify(planner: Level1Planner, sim, q_start, target):
    """Plan, verify in PyBullet, and re-plan once with traj.retry_refine_steps if verification fails."""
    c = planner.c
    Q, info = planner.plan(q_start, target)
    res = verify_trajectory(sim, Q, target, c["success_tol"])
    info.update(retried=False, first_success=res["success"])
    if not res["success"]:
        Q2, info2 = planner.plan(q_start, target, refine_steps=c["retry_refine_steps"])
        res2 = verify_trajectory(sim, Q2, target, c["success_tol"])
        info2["plan_time"] += info["plan_time"]
        info, Q, res = info2, Q2, res2
        info.update(retried=True, first_success=False)
    info.update(res)
    return Q, info


# ------------------------------------------------------------------ classical baseline
def ik_baseline(sim, q_start, target, T: int, max_iters: int = 200, residual: float = 1e-5) -> np.ndarray:
    """Classical baseline: PyBullet null-space IK (rest pose = q_start) + straight joint-space line, (T, 7)."""
    q_start = np.asarray(q_start, float)
    sim.set_q(q_start)
    pos, _ = sim.base_pose()
    target_w = pos + sim.base_rotation() @ np.asarray(target, float)
    lo, hi = sim.lower, sim.upper
    q_goal = p.calculateInverseKinematics(sim.robot, sim.tcp_link, target_w.tolist(), lowerLimits=lo.tolist(),
                                          upperLimits=hi.tolist(), jointRanges=(hi - lo).tolist(),
                                          restPoses=q_start.tolist(), maxNumIterations=max_iters,
                                          residualThreshold=residual, physicsClientId=sim.cid)
    q_goal = np.clip(np.asarray(q_goal, float)[: len(q_start)], lo, hi)
    s = np.linspace(0.0, 1.0, int(T))[:, None]
    return q_start + s * (q_goal - q_start)


# ------------------------------------------------------------------ free-floating execution
def execute_open_loop(sim, Q, target) -> dict:
    """Play a body-frame joint path open-loop on the free-floating spacecraft (joint tracking at the
    control rate, joint speeds at most v_max) and measure how far the TCP misses the target, which is
    fixed in the inertial frame (= body frame at t = 0). The base reacts and nobody corrects for it."""
    cfg = sim.cfg
    hz = cfg["sim"]["control_hz"]
    n_sub = int(round(cfg["sim"]["physics_hz"] / hz))
    dt, v = 1.0 / hz, sim.max_velocity
    Q = np.asarray(Q, float)
    sim.reset(q=Q[0])                                         # base at the origin, identity, at rest
    target_w = np.asarray(target, float)
    _, orn0 = sim.base_pose()
    seg = np.abs(np.diff(Q, axis=0)).max(1) / v               # time per segment at the joint speed limit
    t_knots = np.concatenate([[0.0], np.cumsum(seg)])
    duration = float(t_knots[-1])
    n_steps = int(np.ceil(duration / dt)) + 2 * hz            # + 2 s to settle on the last waypoint
    for k in range(1, n_steps + 1):
        t = min(k * dt, duration)
        q_ref = np.array([np.interp(t, t_knots, Q[:, j]) for j in range(Q.shape[1])])
        sim.apply_joint_velocities((q_ref - sim.get_q()) / dt)
        sim.step(n_sub)
    sim.apply_joint_velocities(np.zeros(Q.shape[1]))
    _, orn1 = sim.base_pose()
    miss = float(np.linalg.norm(sim.tcp_world() - target_w))
    body_err = float(np.linalg.norm(sim.tcp_body() - target_w))
    rot = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(orn0, orn1)))))))
    return {"inertial_miss": miss, "body_reach_err": body_err, "base_rotation_deg": rot, "duration_s": duration,
            "joint_err": float(np.abs(sim.get_q() - Q[-1]).max())}
