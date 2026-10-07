"""Level-1 stage-by-stage evaluation (v1.1): what each planner stage adds.

Same query sets as scripts/eval_level1.py (200 random + 100 hard, same seeds), every result checked in PyBullet
(clearance > 0 at 100 points, reach < 2 cm, within limits):

  baseline         PyBullet null-space IK + straight joint line (classical reference)
  network          TrajNet alone (one forward pass)
  network+refine   + 30 Adam steps on the path's 35 numbers (DistanceNet clearance, FK reach, smoothness)
  network+polish   + 10 damped-least-squares IK steps on the goal pose (no refine)
  refine+polish    both, no PyBullet check (what `plan()` returns)
  full             refine + polish + PyBullet verification, one retry with 100 refine steps

Also saves one worked example (a hard query): the path, its control points and the true clearance along it
after each stage.

  reports/level1/stages.json, reports/level1/stages.md

    python scripts/eval_level1_stages.py [--n-random 200] [--n-hard 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from spacearm.config import ROOT, load_config
from spacearm.datagen import load_dataset
from spacearm.kinematics import ArmKinematics
from spacearm.models.distance_net import DistanceNet
from spacearm.models.traj_net import TrajNet
from spacearm.planner import Level1Planner, free_pool, ik_baseline, plan_and_verify, sample_queries, verify_trajectory
from spacearm.sim import SpaceRobotSim

STAGES = ["baseline", "network", "network+refine", "network+polish", "refine+polish", "full"]
LABELS = {"baseline": "Baseline: PyBullet IK + straight line", "network": "TrajNet alone",
          "network+refine": "TrajNet + refine", "network+polish": "TrajNet + IK polish",
          "refine+polish": "TrajNet + refine + polish", "full": "Full planner (+ PyBullet check, retry)"}
PLAN_ARGS = {"network": (0, False), "network+refine": (None, False), "network+polish": (0, True),
             "refine+polish": (None, True)}


def clearance_along(sim, Q) -> np.ndarray:
    """True signed clearance (m) at every point of a joint path (PyBullet)."""
    out = np.empty(len(Q))
    for k, q in enumerate(Q):
        sim.set_q(q)
        out[k] = min(sim.min_distances())
    return out


def run_stage(stage, planner, sim, q, t, cfg):
    c = cfg["traj"]
    t0 = time.perf_counter()
    if stage == "baseline":
        Q, info = ik_baseline(sim, q, t, c["t_verify"]), {}
        info["plan_time"] = time.perf_counter() - t0
    elif stage == "full":
        Q, info = plan_and_verify(planner, sim, q, t)
        info["plan_time"] = time.perf_counter() - t0                # incl. PyBullet checks and any retry
        return Q, info
    else:
        steps, polish = PLAN_ARGS[stage]
        Q, info = planner.plan(q, t, refine_steps=steps, polish=polish)
    info.update(verify_trajectory(sim, Q, t, c["success_tol"]))
    return Q, info


def summarise(rows: list[dict]) -> dict:
    return {"n": len(rows), "success": float(np.mean([r["success"] for r in rows])),
            "collision_rate": float(np.mean([not r["collision_free"] for r in rows])),
            "reach_ok": float(np.mean([r["reach_err"] < 0.02 for r in rows])),
            "median_reach_err_cm": float(np.median([r["reach_err"] for r in rows]) * 100),
            "median_min_clearance_cm": float(np.median([r["min_dist"] for r in rows]) * 100),
            "median_time_ms": float(np.median([r["plan_time"] for r in rows]) * 1000),
            "p95_time_ms": float(np.percentile([r["plan_time"] for r in rows], 95) * 1000)}


def markdown(res: dict) -> str:
    lines = ["| Set | Stage | Success | Collisions | Reach < 2 cm | Median reach error | Median min. clearance | "
             "Median time |", "|---|---|---|---|---|---|---|---|"]
    for set_name in ("random", "hard"):
        for st in STAGES:
            s = res[set_name][st]
            lines.append(f"| {set_name} ({s['n']}) | {LABELS[st]} | {100 * s['success']:.1f} % | "
                         f"{100 * s['collision_rate']:.1f} % | {100 * s['reach_ok']:.1f} % | "
                         f"{s['median_reach_err_cm']:.2f} cm | {s['median_min_clearance_cm']:.1f} cm | "
                         f"{s['median_time_ms']:.0f} ms |")
    return "\n".join(lines) + "\n\nEvery result is checked in PyBullet: success = no contact at 100 path points, " \
                              "reach < 2 cm, within joint limits. Times on 1 CPU thread.\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-hard", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    c, seed = cfg["traj"], cfg["seed"]
    torch.set_num_threads(1)
    models_dir = ROOT / cfg["paths"]["models_dir"]
    try:
        test_data = load_dataset(ROOT / cfg["paths"]["data_dir"] / "kinematics_test.npz")
    except FileNotFoundError:
        sys.exit("No dataset: run  python scripts/gen_data.py")
    sim, kin = SpaceRobotSim(cfg), ArmKinematics(cfg)
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(models_dir / "distance_net.pt", map_location="cpu"))
    tnet = TrajNet.from_config(cfg, kin.lower, kin.upper)
    tnet.load_state_dict(torch.load(models_dir / "traj_net.pt", map_location="cpu"))
    planner = Level1Planner(kin, dnet, tnet, cfg)
    pool = free_pool(test_data, cfg["collision"]["plan_margin"])

    # Identical query sets to eval_level1.py
    rq, rt, _ = sample_queries(pool, kin, args.n_random, np.random.default_rng(seed + 10), c["min_target_dist"])
    hq, ht, hrng = [], [], np.random.default_rng(seed + 20)
    while len(hq) < args.n_hard:
        q, t, _ = sample_queries(pool, kin, 1, hrng, c["min_target_dist"])
        if not verify_trajectory(sim, ik_baseline(sim, q[0], t[0], c["t_verify"]), t[0], c["success_tol"])["collision_free"]:
            hq.append(q[0])
            ht.append(t[0])
    sets = {"random": (rq, rt), "hard": (np.array(hq), np.array(ht))}

    res, per_query = {}, {}
    for set_name, (Qs, Ts) in sets.items():
        res[set_name], per_query[set_name] = {}, {}
        for st in STAGES:
            rows = [run_stage(st, planner, sim, Qs[k], Ts[k], cfg)[1] for k in range(len(Qs))]
            res[set_name][st] = summarise(rows)
            per_query[set_name][st] = {"success": [bool(r["success"]) for r in rows],
                                       "min_clearance_cm": [round(100 * r["min_dist"], 3) for r in rows],
                                       "reach_err_cm": [round(100 * r["reach_err"], 4) for r in rows],
                                       "time_ms": [round(1000 * r["plan_time"], 2) for r in rows]}
            s = res[set_name][st]
            print(f"{set_name:6s} {st:15s} success {100 * s['success']:5.1f} %  collisions {100 * s['collision_rate']:5.1f} %"
                  f"  reach {s['median_reach_err_cm']:6.2f} cm  time {s['median_time_ms']:6.0f} ms", flush=True)

    # Worked example: the first hard query where TrajNet alone collides and refinement fixes it.
    ex = None
    for k in range(len(hq)):
        if (not per_query["hard"]["network"]["success"][k] and per_query["hard"]["network"]["min_clearance_cm"][k] < 0
                and per_query["hard"]["refine+polish"]["success"][k]):
            ex = k
            break
    if ex is not None:
        q, t = hq[ex], ht[ex]
        example = {"index": ex, "q_start": q.tolist(), "target": t.tolist(), "stages": {}}
        for st in ("baseline", "network", "network+refine", "refine+polish"):
            Q, info = run_stage(st, planner, sim, q, t, cfg)
            example["stages"][st] = {"path": np.round(Q, 5).tolist(),
                                     "clearance_cm": np.round(100 * clearance_along(sim, Q), 3).tolist(),
                                     "reach_err_cm": 100 * info["reach_err"], "success": bool(info["success"]),
                                     "control_points": (np.round(info["control_points"], 5).tolist()
                                                        if "control_points" in info else None)}
        res["example"] = example
    res["per_query"] = per_query
    res["settings"] = {"seed": seed, "threads": 1, "pool": len(pool), "refine_steps": c["refine_steps"],
                       "polish_iters": c["polish_iters"]}
    sim.close()

    out = ROOT / "reports" / "level1"
    (out / "stages.json").write_text(json.dumps(res), encoding="utf-8")
    (out / "stages.md").write_text(markdown(res), encoding="utf-8")
    print("\n" + markdown(res))


if __name__ == "__main__":
    main()
