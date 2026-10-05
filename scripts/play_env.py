"""Run prior-only episodes (zero residual) in the Level-2 environment, print each episode's outcome,
save summary statistics and render the first episode(s) offscreen.

  reports/level2/prior_d<d>.json          success / collision / cost / timing over the episodes
  reports/level2/play_d<d>_ep<k>.mp4       offscreen video (10 fps)
  reports/level2/play_d<d>_ep<k>.png       frame strip (for a quick look without a video player)

    python scripts/play_env.py --difficulty 1.0 --episodes 3 [--render 1] [--seed 0] [--config ...]
    python scripts/play_env.py --gui ...    (workstation only: PyBullet window instead of offscreen frames)
"""
from __future__ import annotations

import argparse
import json
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spacearm.config import ROOT, load_config  # noqa: E402
from spacearm.envs.space_reach_env import SpaceReachEnv  # noqa: E402

STRIP_FRAMES = 6
INK = "#0b0b0b"


def save_strip(frames: list[np.ndarray], titles: list[str], path) -> None:
    idx = np.linspace(0, len(frames) - 1, min(STRIP_FRAMES, len(frames))).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 2.8))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(frames[i])
        ax.set_title(titles[i], fontsize=8, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="episode k uses seed + k")
    ap.add_argument("--render", type=int, default=1, help="number of episodes to render (0 = none)")
    ap.add_argument("--gui", action="store_true", help="PyBullet GUI (workstation only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = SpaceReachEnv(cfg, difficulty=args.difficulty, gui=args.gui)
    out = ROOT / "reports" / "level2"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"d{args.difficulty:g}"
    zero = np.zeros((1, 7), np.float32)
    eps, step_times = [], []
    for k in range(args.episodes):
        _, info = env.reset(seed=args.seed + k)
        d0 = info["d_tcp"]
        render = (not args.gui) and k < args.render
        frames, titles = [], []
        ret = cost = 0.0
        min_clear, obstacle = np.inf, False
        while True:
            if render:
                frames.append(env.render())
                titles.append(f"t={env.t / 10:.1f}s d={info['d_tcp'] * 100:.0f}cm clr={info['clearance'] * 100:.0f}cm")
            t0 = time.perf_counter()
            _, r, te, tr, info = env.step(zero)
            step_times.append(time.perf_counter() - t0)
            ret += r
            cost += info["cost"]
            min_clear = min(min_clear, info["clearance"])
            obstacle |= info["obstacle_active"]
            if args.gui:
                time.sleep(env.dt)
            if te or tr:
                break
        ep = {"seed": args.seed + k, "success": info["success"], "collision": info["collision"], "cost": cost,
              "return": ret, "length": info["t"], "d_start_cm": d0 * 100, "d_final_cm": info["d_tcp"] * 100,
              "min_clearance_cm": min_clear * 100, "obstacle": obstacle, "base_rotation_deg": info["base_rotation_deg"]}
        eps.append(ep)
        outcome = "SUCCESS" if ep["success"] else ("COLLISION" if ep["collision"] else "timeout")
        print(f"ep {k:3d}  {outcome:9s}  steps {ep['length']:3d}  d {ep['d_start_cm']:5.1f} -> {ep['d_final_cm']:5.1f} cm  "
              f"cost {cost:4.0f}  min clearance {ep['min_clearance_cm']:5.1f} cm  obstacle {'yes' if obstacle else 'no '}  "
              f"base rot {ep['base_rotation_deg']:4.1f} deg", flush=True)
        if render:
            frames.append(env.render())
            titles.append(f"end: {outcome.lower()}")
            imageio.mimsave(out / f"play_{tag}_ep{k}.mp4", frames, fps=10, macro_block_size=8)
            save_strip(frames, titles, out / f"play_{tag}_ep{k}.png")
    env.close()

    def rate(key, subset=None):
        sel = [e for e in eps if subset is None or subset(e)]
        return float(np.mean([e[key] for e in sel])) if sel else float("nan")

    summary = {"difficulty": args.difficulty, "episodes": len(eps), "seed": args.seed,
               "success": rate("success"), "collision": rate("collision"), "mean_cost": rate("cost"),
               "mean_length": rate("length"), "obstacle_share": rate("obstacle"),
               "success_with_obstacle": rate("success", lambda e: e["obstacle"]),
               "collision_with_obstacle": rate("collision", lambda e: e["obstacle"]),
               "step_ms_median": float(np.median(step_times) * 1e3), "step_ms_p95": float(np.percentile(step_times, 95) * 1e3),
               "episodes_detail": eps}
    (out / f"prior_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nprior only, d = {args.difficulty:g}, {len(eps)} episodes: success {100 * summary['success']:.1f} %, "
          f"collisions {100 * summary['collision']:.1f} %, mean cost {summary['mean_cost']:.2f}, "
          f"obstacles in {100 * summary['obstacle_share']:.0f} % of episodes, env step {summary['step_ms_median']:.2f} ms (median)")
    print(f"saved {out / f'prior_{tag}.json'}" + (f" and renders play_{tag}_ep*.mp4/.png" if args.render and not args.gui else ""))


if __name__ == "__main__":
    main()
