"""Measure the Phase 1 numbers behind CHECKLIST / DESIGN §8 and save them as JSON.

FK vs PyBullet, free-floating CoM drift and base reaction, analytic capsules vs PyBullet distances,
"ghost" obstacles (no contact forces), worst distance of every allowed-collision-matrix pair,
distance-query speed and the uniform collision share.

    python scripts/check_world_model.py [--config ...] [--seed 0] [--n-acm 20000]
    -> reports/phase1/world_model_check.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from spacearm.config import ROOT, load_config
from spacearm.kinematics import ArmKinematics
from spacearm.robot_model import ARM_LINKS, TCP_LINK
from spacearm.sim import SpaceRobotSim


def segment_distance(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    s = np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(c - (a + s * ab)))


def quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-acm", type=int, default=20000, help="random configurations for the ACM check")
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase1" / "world_model_check.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"] if args.seed is None else args.seed)
    sim = SpaceRobotSim(cfg)
    kin = ArmKinematics(cfg, dtype=torch.float64)
    substeps = int(round(cfg["sim"]["physics_hz"] / cfg["sim"]["control_hz"]))
    res = {}

    # FK vs PyBullet link frames (base at identity -> world == body frame)
    err = 0.0
    for _ in range(500):
        q = rng.uniform(sim.lower, sim.upper)
        sim.set_q(q)
        _, t = kin.forward(torch.tensor(q))
        for k, name in enumerate(ARM_LINKS + [TCP_LINK]):
            err = max(err, float(np.abs(t[k].numpy() - sim.link_frame_world(sim.link_index[name])[0]).max()))
    res["fk_max_error_m"] = err

    # Free-floating: same 2 s motion as test_p1 (CoM must stay put, base must rotate)
    sim.reset()
    com0, (_, orn0) = sim.system_com(), sim.base_pose()
    v = np.array([0.5, -0.4, 0.3, 0.5, -0.5, 0.4, 0.5])
    for _ in range(2 * cfg["sim"]["control_hz"]):
        sim.apply_joint_velocities(v)
        sim.step(substeps)
    res["com_drift_m"] = float(np.linalg.norm(sim.system_com() - com0))
    res["base_rotation_deg"] = quat_angle_deg(orn0, sim.base_pose()[1])
    res["joint_tracking_error_rad"] = float(np.abs(sim.get_q() - 2.0 * v).max())

    # Analytic sphere-capsule distance vs PyBullet (validates the URDF capsule convention)
    sim.reset()
    err = 0.0
    for _ in range(200):
        q = rng.uniform(sim.lower * 0.9, sim.upper * 0.9)
        sim.set_q(q)
        c, r = sim.tcp_world() + rng.normal(0.0, 0.15, 3), 0.08
        body = sim.add_sphere(c, r)
        d_pb = sim.body_distance(body, max_dist=2.0)[0]
        a, b, rad = (x.numpy() for x in kin.capsules(torch.tensor(q)))
        d_an = min(segment_distance(c, a[i], b[i]) - rad[i] - r for i in range(7))
        sim.remove_body(body)
        err = max(err, abs(d_an - d_pb))
    res["capsule_vs_pybullet_max_error_m"] = err

    # Ghost obstacle: sweep link4 through a sphere; joints must keep tracking and momentum must be conserved
    sim.reset()
    body = sim.add_sphere(sim.link_frame_world(sim.link_index["link4"])[0] + [0.12, 0.0, 0.0], 0.1)
    com0, dmin = sim.system_com(), np.inf
    for _ in range(20):
        sim.apply_joint_velocities([0, 0.5, 0, 0, 0, 0, 0])
        sim.step(substeps)
        dmin = min(dmin, sim.body_distance(body)[0])
    res["ghost_sweep"] = {"min_distance_m": float(dmin), "com_drift_m": float(np.linalg.norm(sim.system_com() - com0)),
                          "q2_after_2s": float(sim.get_q()[1]), "q2_expected": 1.0}
    sim.remove_body(body)
    sim.reset()

    # Allowed-collision matrix: worst distance of every skipped pair over random configurations
    worst = {f"{a}-{b}": np.inf for a, b in cfg["robot"]["acm_skip"]}
    for _ in range(args.n_acm):
        sim.set_q(rng.uniform(sim.lower, sim.upper))
        p.performCollisionDetection(physicsClientId=sim.cid)
        for a, b in cfg["robot"]["acm_skip"]:
            for pt in p.getClosestPoints(sim.robot, sim.robot, 1.0, linkIndexA=sim.link_index[a],
                                         linkIndexB=sim.link_index[b], physicsClientId=sim.cid):
                worst[f"{a}-{b}"] = min(worst[f"{a}-{b}"], pt[8])
    res["acm_worst_distance_m"] = {k: float(v) for k, v in worst.items()}
    res["acm_n_configs"] = args.n_acm

    # Query speed and uniform collision share (inside limit_frac of each joint range, as in Phase 2)
    frac = cfg["data"]["limit_frac"]
    mid, half = (sim.lower + sim.upper) / 2, (sim.upper - sim.lower) / 2
    Q = rng.uniform(mid - frac * half, mid + frac * half, (5000, 7))
    t0 = time.perf_counter()
    D = []
    for q in Q:
        sim.set_q(q)
        D.append(sim.min_distances())
    res["query_ms_per_config"] = (time.perf_counter() - t0) / len(Q) * 1e3
    D = np.array(D)
    res["uniform_collision_share"] = {"any": float(np.mean(D.min(1) <= 0)), "body": float(np.mean(D[:, 0] <= 0)),
                                      "self": float(np.mean(D[:, 1] <= 0)), "n": len(Q)}
    res["n_pairs"] = {"body": len(sim.pairs_body), "self": len(sim.pairs_self)}
    res["total_mass_kg"] = float(sum(sim.masses.values()))
    sim.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
