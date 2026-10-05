"""Train the residual PPO-Lagrangian policy on the Level-2 arm task (DESIGN §6.3), resumable and time-boxed.

  models/ppo_lag<tag>.pt              best policy by periodic deterministic evaluation (full difficulty)
  models/ppo_lag<tag>_ckpt.pt         full training state (every --ckpt-every updates and on exit)
  reports/ppo_lag<tag>/progress.csv   one row per update      eval.csv  one row per evaluation
  reports/ppo_lag<tag>/curves.png     success / cost / lambda / collisions vs steps; summary.json

Cloud protocol (CLAUDE.md §6): run in segments and commit the checkpoint + CSV after each one:
    python scripts/train_ppo_lag.py --n-envs 4 --resume --max-minutes 8.5
Smoke run:  python scripts/train_ppo_lag.py --n-envs 4 --steps 200000 --tag smoke --fresh
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spacearm.config import ROOT, load_config  # noqa: E402
from spacearm.envs.space_reach_env import SpaceReachEnv  # noqa: E402
from spacearm.rl.ppo_lag import evaluate_vec, train  # noqa: E402
from spacearm.rl.vec_env import SubprocVecEnv  # noqa: E402

EVAL_SEED = 20_000            # selection seeds; Phase 7's final evaluation uses different seeds
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


class ZeroPolicy:
    """Zero residual = the classical prior alone (reference line)."""

    def act(self, obs):
        obs = np.asarray(obs)
        return np.zeros((*obs.shape[:-1], 7), np.float32)


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) if v not in ("", "nan") else np.nan for k, v in r.items()} for r in csv.DictReader(f)]


def better(res: dict, best: dict | None) -> bool:
    """Success up without more collisions, or same success with fewer collisions."""
    if best is None:
        return True
    return ((res["success"] > best["success"] and res["collision"] <= best["collision"])
            or (res["success"] >= best["success"] and res["collision"] < best["collision"]))


def plot(rep: Path, cost_limit: float, prior: dict | None) -> None:
    rows, evals = read_csv(rep / "progress.csv"), read_csv(rep / "eval.csv")
    if not rows:
        return
    st = np.array([r["steps"] for r in rows]) / 1e6

    def smooth(key, k=10):
        x = np.array([r[key] for r in rows], float)
        out = np.full_like(x, np.nan)
        for i in range(len(x)):
            w = x[max(0, i - k + 1): i + 1]
            out[i] = np.nanmean(w) if np.any(np.isfinite(w)) else np.nan
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_MUTED)
        ax.grid(color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_xlabel("environment steps (millions)", color=INK_MUTED)
    a = axes[0, 0]
    a.plot(st, smooth("success"), color=BLUE, lw=2, label="training episodes (current difficulty)")
    if evals:
        a.plot([e["steps"] / 1e6 for e in evals], [e["success"] for e in evals], color=ORANGE, lw=2, marker="o", ms=4,
               label="deterministic eval, d = 1 (20 episodes)")
    if prior:
        a.axhline(prior["success"], color=INK_MUTED, ls="--", lw=1)
        a.text(st[-1], prior["success"], "prior only (eval)", color=INK_MUTED, ha="right", va="bottom", fontsize=9)
    a.set_ylim(0, 1)
    a.set_title("Success rate", color=INK)
    a.legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower right")
    a = axes[0, 1]
    a.plot(st, smooth("cost"), color=BLUE, lw=2)
    a.axhline(cost_limit, color=INK_MUTED, ls="--", lw=1)
    a.text(st[-1], cost_limit, "cost limit", color=INK_MUTED, ha="right", va="bottom", fontsize=9)
    a.set_title("Near-miss cost per episode (training)", color=INK)
    a = axes[1, 0]
    a.plot(st, [r["lambda"] for r in rows], color=BLUE, lw=2, label="lambda")
    a.plot(st, [r["difficulty"] for r in rows], color=ORANGE, lw=2, label="curriculum difficulty")
    a.set_title("Lagrange multiplier and curriculum", color=INK)
    a.legend(frameon=False, labelcolor=INK, fontsize=8)
    a = axes[1, 1]
    a.plot(st, smooth("collision"), color=BLUE, lw=2, label="training episodes")
    if evals:
        a.plot([e["steps"] / 1e6 for e in evals], [e["collision"] for e in evals], color=ORANGE, lw=2, marker="o", ms=4,
               label="deterministic eval, d = 1")
    a.set_ylim(bottom=0)
    a.set_title("Collision rate", color=INK)
    a.legend(frameon=False, labelcolor=INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "curves.png", dpi=100, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None, help="parallel envs (default ppo.n_envs); batch size is kept")
    ap.add_argument("--steps", type=int, default=None, help="total environment steps (default ppo.total_steps)")
    ap.add_argument("--vec", choices=["subproc", "sync"], default="subproc")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint if it exists")
    ap.add_argument("--fresh", action="store_true", help="start over (deletes this tag's checkpoint and logs)")
    ap.add_argument("--max-minutes", type=float, default=None, help="time box for this segment")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4, help="PyTorch threads in the learner process")
    ap.add_argument("--tag", default="", help="suffix for output names (e.g. smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["ppo"]
    seed = cfg["seed"] if args.seed is None else args.seed
    n_envs = args.n_envs or p["n_envs"]
    n_steps = p["n_envs"] * p["n_steps"] // n_envs              # keep the batch size (2048) for any n_envs
    total = args.steps or p["total_steps"]
    torch.set_num_threads(args.threads)
    tag = f"_{args.tag}" if args.tag else ""
    models = ROOT / cfg["paths"]["models_dir"]
    rep = ROOT / "reports" / f"ppo_lag{tag}"
    best_path, ckpt_path = models / f"ppo_lag{tag}.pt", models / f"ppo_lag{tag}_ckpt.pt"
    models.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for f in (ckpt_path, best_path, rep / "progress.csv", rep / "eval.csv"):
            f.unlink(missing_ok=True)
    if ckpt_path.exists() and not args.resume:
        raise SystemExit(f"{ckpt_path.name} exists: pass --resume to continue or --fresh to start over")

    resume, extra = None, {"best": None, "prior": None, "beats_prior": False}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        resume, extra = ck["train"], ck["extra"]
        total = int(ck["train"]["total_steps"])
        print(f"resuming at update {resume['update']}, {resume['steps']} / {total} steps, lambda {resume['lambda']:.2f}")
        if resume["steps"] >= total:
            print("training already finished")
            plot(rep, p["cost_limit"], extra["prior"])
            return

    make_env = functools.partial(SpaceReachEnv, cfg, 0.0)
    eval_env = functools.partial(SpaceReachEnv, cfg, 1.0)
    evenv = SubprocVecEnv([eval_env] * n_envs)
    t0 = time.perf_counter()
    if extra["prior"] is None:
        extra["prior"] = evaluate_vec(ZeroPolicy(), evenv, args.eval_episodes, EVAL_SEED)
        print(f"prior only (eval, d = 1, {args.eval_episodes} ep): success {extra['prior']['success']:.2f} "
              f"collisions {extra['prior']['collision']:.2f} cost {extra['prior']['cost']:.2f}")
    state = {"ckpt": None}

    def save_ckpt():
        if state["ckpt"] is not None:
            torch.save({"train": state["ckpt"](), "extra": extra}, ckpt_path)

    def callback(ctx) -> bool:
        state["ckpt"] = ctx["checkpoint"]
        append_csv(rep / "progress.csv", ctx["history"][-1])
        u = ctx["update"]
        if u % args.eval_every == 0 or ctx["steps"] >= ctx["total_steps"]:
            res = evaluate_vec(ctx["agent"], evenv, args.eval_episodes, EVAL_SEED)
            improved = better(res, extra["best"])
            if improved:
                extra["best"] = {**res, "update": u, "steps": ctx["steps"]}
                ctx["agent"].save(best_path)
            pr = extra["prior"]
            extra["beats_prior"] |= bool(res["success"] > pr["success"] and res["collision"] <= pr["collision"])
            append_csv(rep / "eval.csv", {"update": u, "steps": ctx["steps"], "success": res["success"],
                                          "collision": res["collision"], "cost": res["cost"], "return": res["return"],
                                          "prior_success": pr["success"], "prior_collision": pr["collision"],
                                          "saved": int(improved)})
            print(f"  eval @ {ctx['steps']}: success {res['success']:.2f} collisions {res['collision']:.2f} "
                  f"cost {res['cost']:.2f} (prior {pr['success']:.2f}/{pr['collision']:.2f})"
                  f"{'  -> saved best' if improved else ''}", flush=True)
        if u % args.ckpt_every == 0:
            save_ckpt()
        return False

    try:
        train(make_env, cfg, total_steps=total, n_envs=n_envs, seed=seed, vec=args.vec, curriculum=True,
              log_fn=lambda s: print(s, flush=True), callback=callback, resume=resume, max_minutes=args.max_minutes,
              n_steps=n_steps)
    finally:
        save_ckpt()
        evenv.close()
        plot(rep, p["cost_limit"], extra["prior"])
        rows = read_csv(rep / "progress.csv")
        summary = {"steps": int(rows[-1]["steps"]) if rows else 0, "total_steps": total,
                   "updates": int(rows[-1]["update"]) if rows else 0, "best_eval": extra["best"],
                   "prior_eval": extra["prior"], "beats_prior": extra["beats_prior"],
                   "segment_minutes": (time.perf_counter() - t0) / 60, "n_envs": n_envs, "n_steps": n_steps,
                   "eval_seed": EVAL_SEED, "eval_episodes": args.eval_episodes}
        (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"segment done: {summary['steps']} / {total} steps in {summary['segment_minutes']:.1f} min; "
              f"best eval {extra['best']['success'] if extra['best'] else float('nan'):.2f} "
              f"(prior {extra['prior']['success']:.2f}); checkpoint {ckpt_path.name}")


if __name__ == "__main__":
    main()
