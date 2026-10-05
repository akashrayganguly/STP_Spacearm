"""Level-2 evaluation (DESIGN §6.6): scenarios x methods on identical fixed-seed episodes, in parallel.

  S1 nominal (d = 0) | S2 obstacles only | S3 faults + noise only | S4 everything (d = 1)
  methods: prior | prior + shield | RL | RL + shield;  ablation (S2, S4): prior with the reflex off (k_obs = 0)
  Episode k of every method uses seed seed_base + k, so all methods face the same starts, targets and surprises.

  reports/level2/eval_<S>.json   per scenario, written as soon as it finishes
  reports/level2/results.md      table over all evaluated scenarios
  reports/level2/margin_sweep_s<seed>_n<N>.json  with --sweep-margins (S4, RL + shield and prior + shield per margin)

    python scripts/eval_level2.py [--scenarios S1 S2 S3 S4] [--episodes 100] [--seed-base 10000] [--workers 4]
    python scripts/eval_level2.py --sweep-margins 0.05 0.06 0.07 0.08 --seed-base 30000
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from spacearm.config import ROOT, deep_update, load_config

SCENARIOS = {
    "S1": {"difficulty": 0.0, "overrides": {}, "label": "S1 nominal"},
    "S2": {"difficulty": 1.0, "overrides": {"env": {"faults": {"enabled": False}, "noise": {"enabled": False}}},
           "label": "S2 obstacles only"},
    "S3": {"difficulty": 1.0, "overrides": {"env": {"obstacle": {"prob": 0.0}}}, "label": "S3 faults + noise only"},
    "S4": {"difficulty": 1.0, "overrides": {}, "label": "S4 everything"},
}
METHODS = {"prior": (False, False, {}), "prior+shield": (False, True, {}), "rl": (True, False, {}),
           "rl+shield": (True, True, {}), "prior (reflex off)": (False, False, {"prior": {"k_obs": 0.0}})}
ABLATION_SCENARIOS = ("S2", "S4")


class _Zero:
    def act(self, obs):
        return np.zeros((*np.asarray(obs).shape[:-1], 7), np.float32)


def _run_chunk(task: dict) -> list[dict]:
    """Worker: build env (+ shield) and policy, run the given seeds; returns per-episode records."""
    import torch

    torch.set_num_threads(1)
    from spacearm.envs.space_reach_env import SpaceReachEnv
    from spacearm.models.distance_net import DistanceNet
    from spacearm.rl.ppo_lag import PPOLagAgent
    from spacearm.safety.shield import shield_from_config

    cfg = deep_update(load_config(task["config"]), task["overrides"])
    env = SpaceReachEnv(cfg, difficulty=task["difficulty"])
    if task["shield"]:
        dnet = DistanceNet.from_config(cfg)
        dnet.load_state_dict(torch.load(ROOT / cfg["env"]["distance_net_path"], map_location="cpu"))
        env.set_shield(shield_from_config(env.kin, cfg, dnet))
    policy = PPOLagAgent.load(ROOT / task["policy"]) if task["policy"] else _Zero()
    out = []
    for seed in task["seeds"]:
        obs, info = env.reset(seed=seed)
        cost, n, t_pol, t_step, obstacle = 0.0, 0, 0.0, 0.0, False
        min_clear = np.inf
        while True:
            t0 = time.perf_counter()
            a = policy.act(obs["actor"])
            t1 = time.perf_counter()
            obs, r, te, tr, info = env.step(a)
            t_pol += t1 - t0
            t_step += time.perf_counter() - t1
            cost += info["cost"]
            n += 1
            min_clear = min(min_clear, info["clearance"])
            obstacle |= info["obstacle_active"]
            if te or tr:
                break
        out.append({"seed": seed, "success": bool(info["success"]), "collision": bool(info["collision"]), "cost": cost,
                    "length": n, "time_to_reach_s": n * env.dt if info["success"] else None,
                    "base_rotation_deg": info["base_rotation_deg"], "min_clearance_cm": 100 * min_clear,
                    "obstacle": obstacle, "shield_interventions": info.get("shield_interventions", 0),
                    "policy_ms": 1e3 * t_pol / n, "env_step_ms": 1e3 * t_step / n})
    env.close()
    return out


def summarize(eps: list[dict]) -> dict:
    tt = [e["time_to_reach_s"] for e in eps if e["time_to_reach_s"] is not None]
    return {"n": len(eps), "success": float(np.mean([e["success"] for e in eps])),
            "collision": float(np.mean([e["collision"] for e in eps])),
            "mean_cost": float(np.mean([e["cost"] for e in eps])),
            "time_to_reach_s": float(np.mean(tt)) if tt else float("nan"),
            "base_rotation_deg": float(np.mean([e["base_rotation_deg"] for e in eps])),
            "shield_interventions_per_episode": float(np.mean([e["shield_interventions"] for e in eps])),
            "shield_intervention_rate": float(np.sum([e["shield_interventions"] for e in eps]) /
                                              max(1, np.sum([e["length"] for e in eps]))),
            "policy_ms_per_step": float(np.mean([e["policy_ms"] for e in eps])),
            "env_step_ms": float(np.mean([e["env_step_ms"] for e in eps]))}


def run(pool, scenario: str, method: str, seeds: list[int], workers: int, config, policy_path: str,
        extra_overrides: dict | None = None) -> tuple[dict, list[dict]]:
    use_rl, shield, method_over = METHODS[method]
    sc = SCENARIOS[scenario]
    over = deep_update(deep_update(sc["overrides"], method_over), extra_overrides or {})
    chunks = [seeds[i::workers] for i in range(workers)]
    tasks = [{"config": config, "overrides": over, "difficulty": sc["difficulty"], "shield": shield,
              "policy": policy_path if use_rl else None, "seeds": ch} for ch in chunks if ch]
    eps = sorted([e for part in pool.map(_run_chunk, tasks) for e in part], key=lambda e: e["seed"])
    return summarize(eps), eps


def table(results: dict) -> str:
    lines = ["| Scenario | Method | Success | Collisions | Mean cost | Time to reach | Base rotation | Shield interventions |",
             "|---|---|---|---|---|---|---|---|"]
    for sname, res in results.items():
        for m, s in res["methods"].items():
            iv = f"{100 * s['shield_intervention_rate']:.1f} % of steps" if "shield" in m else "—"
            lines.append(f"| {SCENARIOS[sname]['label']} | {m} | {100 * s['success']:.0f} % | {100 * s['collision']:.0f} % | "
                         f"{s['mean_cost']:.2f} | {s['time_to_reach_s']:.1f} s | {s['base_rotation_deg']:.1f}° | {iv} |")
    n = next(iter(results.values()))["episodes"]
    return "\n".join(lines) + f"\n\n{n} identical fixed-seed episodes per cell (seeds {next(iter(results.values()))['seed_base']}+).\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policy", default="models/ppo_lag.pt")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--sweep-margins", type=float, nargs="+", default=None,
                    help="S4 only: evaluate shielded methods for these shield.self_margin values")
    args = ap.parse_args()

    out = ROOT / "reports" / "level2"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        if args.sweep_margins:
            sweep = {"seed_base": args.seed_base, "episodes": args.episodes, "scenario": "S4", "margins": {}}
            for m in args.sweep_margins:
                over = {"shield": {"self_margin": m}}
                row = {}
                for method in ("prior+shield", "rl+shield"):
                    row[method], _ = run(pool, "S4", method, seeds, args.workers, args.config, args.policy, over)
                sweep["margins"][f"{m:.3f}"] = row
                print(f"self_margin {m * 100:.0f} cm: " + "  |  ".join(
                    f"{k}: success {100 * v['success']:.0f} %, collisions {100 * v['collision']:.0f} %, "
                    f"interventions {100 * v['shield_intervention_rate']:.0f} % of steps" for k, v in row.items()), flush=True)
            name = f"margin_sweep_s{args.seed_base}_n{args.episodes}.json"
            (out / name).write_text(json.dumps(sweep, indent=2), encoding="utf-8")
            print(f"saved {out / name}")
            return

        for sname in args.scenarios:
            t0 = time.perf_counter()
            methods = ["prior", "prior+shield", "rl", "rl+shield"]
            if not args.no_ablation and sname in ABLATION_SCENARIOS:
                methods.append("prior (reflex off)")
            res = {"scenario": sname, "label": SCENARIOS[sname]["label"], "episodes": args.episodes,
                   "seed_base": args.seed_base, "methods": {}, "episodes_detail": {}}
            for m in methods:
                res["methods"][m], res["episodes_detail"][m] = run(pool, sname, m, seeds, args.workers, args.config,
                                                                   args.policy)
                s = res["methods"][m]
                print(f"{sname} {m:20s} success {100 * s['success']:5.1f} %  collisions {100 * s['collision']:4.1f} %  "
                      f"cost {s['mean_cost']:5.2f}  reach {s['time_to_reach_s']:5.1f} s  rot {s['base_rotation_deg']:4.1f} deg",
                      flush=True)
            res["minutes"] = (time.perf_counter() - t0) / 60
            (out / f"eval_{sname}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"saved {out / f'eval_{sname}.json'} ({res['minutes']:.1f} min)", flush=True)

    results = {}
    for sname in SCENARIOS:
        f = out / f"eval_{sname}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            results[sname] = {"methods": r["methods"], "episodes": r["episodes"], "seed_base": r["seed_base"]}
    (out / "results.md").write_text(table(results), encoding="utf-8")
    print("\n" + table(results))


if __name__ == "__main__":
    main()
