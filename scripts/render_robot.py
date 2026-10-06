"""Offscreen snapshots of named arm configurations with their clearances (headless replacement for the GUI).

Writes two PNGs to reports/figures/:
  robot_configs.png  zero, folded-into-deck and seeded random configurations, clearance in each title
  robot_views.png    the zero configuration from the front, the side and above (geometry check)

    python scripts/render_robot.py [--config ...] [--seed 0] [--out reports/figures] [--size 480 360]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402

from spacearm.config import ROOT, load_config  # noqa: E402
from spacearm.sim import SpaceRobotSim  # noqa: E402

# Arm folded back over the payload: upper arm points -x, forearm points down into the deck (as in test_p1).
Q_FOLDED = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0])
CAM_TARGET = [0.0, 0.0, 0.55]


def render(sim: SpaceRobotSim, width: int, height: int, yaw: float, pitch: float, dist: float) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(CAM_TARGET, dist, yaw, pitch, 0.0, 2, physicsClientId=sim.cid)
    proj = p.computeProjectionMatrixFOV(45.0, width / height, 0.05, 20.0, physicsClientId=sim.cid)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                                        lightDirection=[1.0, -1.0, 2.0], shadow=0, physicsClientId=sim.cid)
    return np.reshape(np.asarray(rgba, np.uint8), (height, width, 4))[:, :, :3]


def clearance_title(name: str, sim: SpaceRobotSim) -> str:
    d_body, d_self = sim.min_distances()
    clip = sim.max_query_dist

    def fmt(d: float) -> str:
        return f">={clip * 100:.0f} cm" if d >= clip else f"{d * 100:+.1f} cm"

    flag = "  COLLISION" if min(d_body, d_self) <= 0 else ""
    return f"{name}{flag}\nd_body {fmt(d_body)} | d_self {fmt(d_self)}"


def named_configs(sim: SpaceRobotSim, rng: np.random.Generator, n_free: int = 2, n_hit: int = 2):
    """Zero, folded, then the first n_free clear and n_hit colliding uniform samples of the seeded stream."""
    out = [("zero", np.zeros(7)), ("folded into deck", Q_FOLDED)]
    free, hit = [], []
    while len(free) < n_free or len(hit) < n_hit:
        q = rng.uniform(sim.lower, sim.upper)
        sim.set_q(q)
        bucket = hit if sim.in_collision() else free
        if len(bucket) < (n_hit if bucket is hit else n_free):
            bucket.append(q)
    out += [(f"random clear {k + 1}", q) for k, q in enumerate(free)]
    out += [(f"random colliding {k + 1}", q) for k, q in enumerate(hit)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="random configurations (default: seed from config)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures"))
    ap.add_argument("--size", type=int, nargs=2, default=[480, 360], metavar=("W", "H"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"] if args.seed is None else args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = args.size
    sim = SpaceRobotSim(cfg)                        # DIRECT (headless)
    try:
        configs = named_configs(sim, rng)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
        for ax, (name, q) in zip(axes.flat, configs):
            sim.set_q(q)
            ax.imshow(render(sim, w, h, yaw=40.0, pitch=-22.0, dist=3.3))
            ax.set_title(clearance_title(name, sim), fontsize=10)
            ax.axis("off")
            print(f"{name:20s} q = {np.array2string(np.degrees(q), precision=0, suppress_small=True)} deg"
                  f"  -> {clearance_title('', sim).strip()}")
        fig.suptitle("Arm configurations (body frame; clearance = signed distance, clipped at "
                     f"{sim.max_query_dist * 100:.0f} cm)")
        fig.tight_layout()
        fig.savefig(out / "robot_configs.png", dpi=90)
        plt.close(fig)

        sim.set_q(np.zeros(7))
        views = [("front (from +x)", 0.0 + 90.0, -10.0), ("side (from -y)", 0.0, -10.0), ("top (from +z)", 90.0, -89.0)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, (name, yaw, pitch) in zip(axes, views):
            ax.imshow(render(sim, w, h, yaw=yaw, pitch=pitch, dist=4.2))
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle("Zero configuration: bus (gold), panels (blue), payload (grey), pedestal, arm, TCP (red)")
        fig.tight_layout()
        fig.savefig(out / "robot_views.png", dpi=90)
        plt.close(fig)
        print(f"saved {out / 'robot_configs.png'} and {out / 'robot_views.png'}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
