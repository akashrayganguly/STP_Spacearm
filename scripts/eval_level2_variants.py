"""Level-2 variant study (v1.1): which surprise hurts which method, and how far beyond training it holds up.

  difficulty sweep : everything scaled by d = 0, 0.25, 0.5, 0.75, 1 (as in training)
  obstacle only    : a ball in every episode (faults and noise off): static, drifting, small, large
  faults only      : encoder bias, weak motors, encoder slip (obstacle and noise off)
  noise only       : sensor noise at the training level (obstacle and faults off)
  stress tests     : beyond the training ranges (ball 15-20 cm, drift 6 cm/s, bias 4 deg, noise x3)

Each variant: the same 100 fixed-seed episodes (seeds 40000+, never used for training, model selection or
the main table) for prior | prior + shield | RL | RL + shield. Starts and targets are drawn first from the
seed, so they are also identical across variants; only the surprises differ.

  reports/level2/variants/<name>.json   per variant, written as soon as it finishes (re-runs skip finished ones)
  reports/level2/variants.md            summary table

    python scripts/eval_level2_variants.py [--episodes 100] [--seed-base 40000] [--workers 4] [--only d0.50 bias]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

from eval_level2 import METHODS, _run_chunk, summarize   # same worker and summary as the main evaluation

from spacearm.config import ROOT, deep_update

QUIET = {"faults": {"enabled": False}, "noise": {"enabled": False}}
NO_OBSTACLE = {"obstacle": {"prob": 0.0}}
ALWAYS_OBSTACLE = {"faults": {"enabled": False}, "noise": {"enabled": False}, "obstacle": {"prob": 1.0}}
FAULTS_ONLY = {"obstacle": {"prob": 0.0}, "noise": {"enabled": False}}
NO_FAULT = {"bias_deg": 0.0, "gain_min": 1.0, "slip_prob": 0.0}


def _env(over: dict) -> dict:
    return {"env": over}


VARIANTS = {
    # group, label, difficulty, config overrides (training values are in configs/default.yaml -> env)
    "d0.00": ("difficulty", "d = 0 (nominal)", 0.0, {}),
    "d0.25": ("difficulty", "d = 0.25", 0.25, {}),
    "d0.50": ("difficulty", "d = 0.5", 0.5, {}),
    "d0.75": ("difficulty", "d = 0.75", 0.75, {}),
    "d1.00": ("difficulty", "d = 1 (everything)", 1.0, {}),
    "obs_static": ("obstacle", "static ball, 5-12 cm", 1.0,
                   _env(deep_update(ALWAYS_OBSTACLE, {"obstacle": {"move_prob": 0.0}}))),
    "obs_drift": ("obstacle", "drifting ball, up to 3 cm/s", 1.0,
                  _env(deep_update(ALWAYS_OBSTACLE, {"obstacle": {"move_prob": 1.0}}))),
    "obs_small": ("obstacle", "small ball, 5-7 cm", 1.0,
                  _env(deep_update(ALWAYS_OBSTACLE, {"obstacle": {"radius": [0.05, 0.07]}}))),
    "obs_large": ("obstacle", "large ball, 10-12 cm", 1.0,
                  _env(deep_update(ALWAYS_OBSTACLE, {"obstacle": {"radius": [0.10, 0.12]}}))),
    "bias": ("faults", "encoder bias, up to 2 deg", 1.0,
             _env(deep_update(FAULTS_ONLY, {"faults": {**NO_FAULT, "bias_deg": 2.0}}))),
    "weak": ("faults", "weak motors, 70-100 %", 1.0,
             _env(deep_update(FAULTS_ONLY, {"faults": {**NO_FAULT, "gain_min": 0.7}}))),
    "slip": ("faults", "encoder slip, 3 deg (every episode)", 1.0,
             _env(deep_update(FAULTS_ONLY, {"faults": {**NO_FAULT, "slip_prob": 1.0}}))),
    "noise": ("noise", "sensor noise (training level)", 1.0,
              _env(deep_update(NO_OBSTACLE, {"faults": {"enabled": False}}))),
    "stress_xl_ball": ("stress", "ball 15-20 cm", 1.0,
                       _env(deep_update(ALWAYS_OBSTACLE, {"obstacle": {"radius": [0.15, 0.20]}}))),
    "stress_fast_ball": ("stress", "ball drifting up to 6 cm/s", 1.0,
                         _env(deep_update(ALWAYS_OBSTACLE, {"obstacle": {"move_prob": 1.0, "max_speed": 0.06}}))),
    "stress_bias4": ("stress", "encoder bias up to 4 deg", 1.0,
                     _env(deep_update(FAULTS_ONLY, {"faults": {**NO_FAULT, "bias_deg": 4.0}}))),
    "stress_noise3": ("stress", "sensor noise x3", 1.0,
                      _env(deep_update(deep_update(NO_OBSTACLE, {"faults": {"enabled": False}}),
                                       {"noise": {"encoder_deg": 0.3, "vision_m": 0.015, "gyro": 0.006}}))),
}
EVAL_METHODS = ["prior", "prior+shield", "rl", "rl+shield"]
KEEP = ("seed", "success", "collision", "cost", "length", "min_clearance_cm", "obstacle", "shield_interventions",
        "base_rotation_deg")


def run_variant(pool, name: str, seeds: list[int], workers: int, config, policy: str) -> dict:
    group, label, difficulty, over = VARIANTS[name]
    res = {"variant": name, "group": group, "label": label, "difficulty": difficulty, "overrides": over,
           "episodes": len(seeds), "seed_base": seeds[0], "methods": {}, "episodes_detail": {}}
    t0 = time.perf_counter()
    for m in EVAL_METHODS:
        use_rl, shield, method_over = METHODS[m]
        chunks = [seeds[i::workers] for i in range(workers)]
        tasks = [{"config": config, "overrides": deep_update(over, method_over), "difficulty": difficulty,
                  "shield": shield, "policy": policy if use_rl else None, "seeds": ch} for ch in chunks if ch]
        eps = sorted([e for part in pool.map(_run_chunk, tasks) for e in part], key=lambda e: e["seed"])
        res["methods"][m] = summarize(eps)
        res["episodes_detail"][m] = [{k: e[k] for k in KEEP} for e in eps]
        s = res["methods"][m]
        print(f"{name:17s} {m:13s} success {100 * s['success']:5.1f} %  collisions {100 * s['collision']:4.1f} %  "
              f"cost {s['mean_cost']:5.2f}  reach {s['time_to_reach_s']:5.1f} s", flush=True)
    res["minutes"] = (time.perf_counter() - t0) / 60
    return res


def table(results: dict) -> str:
    lines = ["| Group | Variant | prior | prior + shield | RL | RL + shield |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        cells = [f"{100 * r['methods'][m]['success']:.0f} % / {100 * r['methods'][m]['collision']:.0f} %"
                 for m in EVAL_METHODS]
        lines.append(f"| {r['group']} | {r['label']} | " + " | ".join(cells) + " |")
    first = next(iter(results.values()))
    return ("Success / collisions, " + f"{first['episodes']} identical fixed-seed episodes per cell "
            f"(seeds {first['seed_base']}+).\n\n" + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=40_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policy", default="models/ppo_lag.pt")
    ap.add_argument("--only", nargs="+", default=None, choices=list(VARIANTS))
    ap.add_argument("--redo", action="store_true", help="re-run variants that already have a JSON file")
    args = ap.parse_args()

    out = ROOT / "reports" / "level2" / "variants"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    names = args.only or list(VARIANTS)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for name in names:
            path = out / f"{name}.json"
            if path.exists() and not args.redo:
                print(f"{name}: done already ({path.name})")
                continue
            res = run_variant(pool, name, seeds, args.workers, args.config, args.policy)
            path.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(f"saved {path} ({res['minutes']:.1f} min)", flush=True)

    results = {n: json.loads((out / f"{n}.json").read_text(encoding="utf-8")) for n in VARIANTS
               if (out / f"{n}.json").exists()}
    (ROOT / "reports" / "level2" / "variants.md").write_text(table(results), encoding="utf-8")
    print("\n" + table(results))


if __name__ == "__main__":
    main()
