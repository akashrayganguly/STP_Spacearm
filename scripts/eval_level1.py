"""Level-1 evaluation (DESIGN §5.5): random + hard query sets, 4 methods, exact PyBullet verification,
and an open-loop free-floating check.

  random set : 200 queries from the collision-free pool of the uniform test set (unseen in training)
  hard set   : 100 queries on which the classical baseline collides (sampled until 100 are found)
  methods    : baseline (PyBullet null-space IK + straight joint line) | network only | network + IK polish |
               full planner (refine + polish + verify, one retry with more refinement)
  free floating: 20 successful full plans played open-loop on the floating spacecraft (inertial miss, base rotation)

  reports/level1/results.json, reports/level1/results.md

    python scripts/eval_level1.py [--config ...] [--seed 0] [--n-random 200] [--n-hard 100] [--n-float 20]
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
from spacearm.planner import (Level1Planner, execute_open_loop, free_pool, ik_baseline, plan_and_verify,
                              sample_queries, verify_trajectory)
from spacearm.sim import SpaceRobotSim

METHODS = ["baseline", "network", "network+polish", "full"]
LABELS = {"baseline": "Baseline (PyBullet IK + line)", "network": "Network only",
          "network+polish": "Network + IK polish", "full": "Full planner (refine + polish + verify)"}


def run_method(name, planner, sim, q, t, cfg):
    c = cfg["traj"]
    t0 = time.perf_counter()
    if name == "baseline":
        Q = ik_baseline(sim, q, t, c["t_verify"])
        info = {"plan_time": time.perf_counter() - t0}
    elif name == "network":
        Q, info = planner.plan(q, t, refine_steps=0, polish=False)
    elif name == "network+polish":
        Q, info = planner.plan(q, t, refine_steps=0, polish=True)
    else:
        Q, info = plan_and_verify(planner, sim, q, t)
        info["total_time"] = time.perf_counter() - t0
        return Q, info
    info.update(verify_trajectory(sim, Q, t, c["success_tol"]))
    return Q, info


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows),
           "success": float(np.mean([r["success"] for r in rows])),
           "collision_rate": float(np.mean([not r["collision_free"] for r in rows])),
           "within_limits": float(np.mean([r["within_limits"] for r in rows])),
           "median_reach_err_cm": float(np.median([r["reach_err"] for r in rows]) * 100),
           "median_plan_time_s": float(np.median([r["plan_time"] for r in rows])),
           "p95_plan_time_s": float(np.percentile([r["plan_time"] for r in rows], 95))}
    if "retried" in rows[0]:
        out["retry_rate"] = float(np.mean([r["retried"] for r in rows]))
        out["first_try_success"] = float(np.mean([r["first_success"] for r in rows]))
        out["median_total_time_s"] = float(np.median([r["total_time"] for r in rows]))
    return out


def markdown(res: dict) -> str:
    lines = ["| Set | Method | Success | Collisions | Median reach error | Median plan time |",
             "|---|---|---|---|---|---|"]
    for set_name in ("random", "hard"):
        for m in METHODS:
            s = res[set_name][m]
            lines.append(f"| {set_name} ({s['n']}) | {LABELS[m]} | {100 * s['success']:.1f} % | "
                         f"{100 * s['collision_rate']:.1f} % | {s['median_reach_err_cm']:.2f} cm | "
                         f"{s['median_plan_time_s'] * 1000:.0f} ms |")
    f = res["free_floating"]
    lines += ["", f"Full planner: retry rate {100 * res['random']['full']['retry_rate']:.1f} % (random), "
                  f"{100 * res['hard']['full']['retry_rate']:.1f} % (hard); median time incl. PyBullet verification "
                  f"{res['random']['full']['median_total_time_s'] * 1000:.0f} ms (random).",
              f"Hard set: {res['hard_search']['found']} baseline-colliding queries in {res['hard_search']['tried']} "
              f"random queries ({100 * res['hard_search']['baseline_collision_share']:.1f} %).",
              "", "| Open-loop on the free-floating spacecraft (" + str(f["n"]) + " plans) | median | max |",
              "|---|---|---|",
              f"| Inertial TCP miss | {f['inertial_miss_median_cm']:.1f} cm | {f['inertial_miss_max_cm']:.1f} cm |",
              f"| Base rotation | {f['base_rotation_median_deg']:.1f}° | {f['base_rotation_max_deg']:.1f}° |",
              f"| Body-frame reach error (tracking) | {f['body_reach_err_median_cm']:.2f} cm | "
              f"{f['body_reach_err_max_cm']:.2f} cm |"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-hard", type=int, default=100)
    ap.add_argument("--n-float", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1, help="PyTorch threads (1 = deployment-like timing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["traj"]
    seed = cfg["seed"] if args.seed is None else args.seed
    torch.set_num_threads(args.threads)
    models_dir = ROOT / cfg["paths"]["models_dir"]
    for f in ("distance_net.pt", "traj_net.pt"):
        if not (models_dir / f).exists():
            sys.exit(f"models/{f} missing: run the Phase 3/4 training scripts first")
    try:
        test_data = load_dataset(ROOT / cfg["paths"]["data_dir"] / "kinematics_test.npz")
    except FileNotFoundError:
        sys.exit("No dataset: run  python scripts/gen_data.py")

    sim = SpaceRobotSim(cfg)
    kin = ArmKinematics(cfg)
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(models_dir / "distance_net.pt", map_location="cpu"))
    tnet = TrajNet.from_config(cfg, kin.lower, kin.upper)
    tnet.load_state_dict(torch.load(models_dir / "traj_net.pt", map_location="cpu"))
    planner = Level1Planner(kin, dnet, tnet, cfg)
    pool = free_pool(test_data, cfg["collision"]["plan_margin"])

    # Query sets
    rq, rt, _ = sample_queries(pool, kin, args.n_random, np.random.default_rng(seed + 10), c["min_target_dist"])
    hq, ht, tried = [], [], 0
    hrng = np.random.default_rng(seed + 20)
    t0 = time.perf_counter()
    while len(hq) < args.n_hard:
        q, t, _ = sample_queries(pool, kin, 1, hrng, c["min_target_dist"])
        tried += 1
        if not verify_trajectory(sim, ik_baseline(sim, q[0], t[0], c["t_verify"]), t[0], c["success_tol"])["collision_free"]:
            hq.append(q[0])
            ht.append(t[0])
    print(f"hard set: {len(hq)} baseline-colliding queries out of {tried} ({time.perf_counter() - t0:.0f} s)")
    sets = {"random": (rq, rt), "hard": (np.array(hq), np.array(ht))}

    res, full_paths = {}, []
    for set_name, (Qs, Ts) in sets.items():
        res[set_name] = {}
        for m in METHODS:
            rows = []
            for k in range(len(Qs)):
                Q, info = run_method(m, planner, sim, Qs[k], Ts[k], cfg)
                rows.append(info)
                if m == "full" and set_name == "random" and info["success"]:
                    full_paths.append((Q, Ts[k]))
            res[set_name][m] = summarise(rows)
            s = res[set_name][m]
            print(f"{set_name:6s} {m:15s} success {100 * s['success']:5.1f} %  collisions {100 * s['collision_rate']:5.1f} %"
                  f"  reach {s['median_reach_err_cm']:6.2f} cm  time {1000 * s['median_plan_time_s']:6.0f} ms", flush=True)

    # Open-loop on the free-floating spacecraft
    fl = [execute_open_loop(sim, Q, t) for Q, t in full_paths[: args.n_float]]
    sim.reset()
    res["free_floating"] = {
        "n": len(fl),
        "inertial_miss_median_cm": float(np.median([f["inertial_miss"] for f in fl]) * 100),
        "inertial_miss_max_cm": float(np.max([f["inertial_miss"] for f in fl]) * 100),
        "base_rotation_median_deg": float(np.median([f["base_rotation_deg"] for f in fl])),
        "base_rotation_max_deg": float(np.max([f["base_rotation_deg"] for f in fl])),
        "body_reach_err_median_cm": float(np.median([f["body_reach_err"] for f in fl]) * 100),
        "body_reach_err_max_cm": float(np.max([f["body_reach_err"] for f in fl]) * 100),
        "median_duration_s": float(np.median([f["duration_s"] for f in fl]))}
    res["hard_search"] = {"found": len(hq), "tried": tried, "baseline_collision_share": len(hq) / tried}
    res["settings"] = {"seed": seed, "threads": args.threads, "pool": len(pool)}
    sim.close()

    out = ROOT / "reports" / "level1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    print("\n" + markdown(res))
    print(f"saved {out / 'results.json'} and {out / 'results.md'}")


if __name__ == "__main__":
    main()
