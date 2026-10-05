"""Train the TrajNet (DESIGN §5.3) self-supervised on outcomes: exact FK reach + DistanceNet clearance + smoothness.

Queries: random (start, goal) pairs from the collision-free pool of data/kinematics_dataset.npz
(clearance > collision.plan_margin); target = TCP of the goal sample (always reachable).
Every --eval-every steps the raw network is checked on 512 fixed queries from the test-set pool.

  models/traj_net.pt                  trained weights
  reports/traj_net/progress.csv       loss parts, lr, network-only validation metrics
  reports/traj_net/training.png       curves

    python scripts/train_traj_net.py [--config ...] [--seed 0] [--steps 10000] [--threads 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spacearm.config import ROOT, load_config  # noqa: E402
from spacearm.datagen import load_dataset  # noqa: E402
from spacearm.kinematics import ArmKinematics  # noqa: E402
from spacearm.losses import level1_loss  # noqa: E402
from spacearm.models.distance_net import DistanceNet  # noqa: E402
from spacearm.models.traj_net import TrajNet, curve  # noqa: E402
from spacearm.planner import free_pool, sample_queries  # noqa: E402

N_VAL = 512
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


@torch.no_grad()
def validate(tnet, dnet, kin, qs, tg, cfg) -> dict:
    P = tnet(qs, tg)
    Q = curve(P, cfg["traj"]["t_train"])
    clear = dnet.clearance(Q)
    loss, parts = level1_loss(Q, kin.tcp(P[:, -1]), tg, clear, cfg)
    reach = torch.linalg.norm(kin.tcp(P[:, -1]) - tg, dim=-1)
    return {"val_loss": loss.item(), "val_reach_median_cm": reach.median().item() * 100,
            "val_reach_lt_2cm": (reach < cfg["traj"]["success_tol"]).float().mean().item(),
            "val_pred_collision": (clear.min(1).values < 0).float().mean().item()}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)


def plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    st = [r["step"] for r in rows]
    a = axes[0]
    a.plot(st, [r["loss"] for r in rows], color=BLUE, lw=1.5, label="train (batch)")
    ev = [r for r in rows if r.get("val_loss") not in (None, "")]
    a.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], color=ORANGE, lw=2, label="validation (512 fixed)")
    a.set_yscale("log")
    a.set_title("Level-1 loss (network only)", color=INK)
    a.set_xlabel("step", color=INK_MUTED)
    a.legend(frameon=False, labelcolor=INK)
    a = axes[1]
    a.plot([r["step"] for r in ev], [r["val_reach_median_cm"] for r in ev], color=BLUE, lw=2)
    a.set_yscale("log")
    a.set_title("Validation: median reach error of the raw goal", color=INK)
    a.set_xlabel("step", color=INK_MUTED)
    a.set_ylabel("cm", color=INK_MUTED)
    a = axes[2]
    a.plot([r["step"] for r in ev], [100 * r["val_pred_collision"] for r in ev], color=BLUE, lw=2)
    a.set_title("Validation: paths with predicted clearance < 0", color=INK)
    a.set_xlabel("step", color=INK_MUTED)
    a.set_ylabel("% of queries", color=INK_MUTED)
    a.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="default: traj.steps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["traj"]
    seed = cfg["seed"] if args.seed is None else args.seed
    steps = c["steps"] if args.steps is None else args.steps
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)

    data_dir, models_dir = ROOT / cfg["paths"]["data_dir"], ROOT / cfg["paths"]["models_dir"]
    try:
        train_data, test_data = load_dataset(data_dir / "kinematics_dataset.npz"), load_dataset(data_dir / "kinematics_test.npz")
    except FileNotFoundError:
        sys.exit(f"No dataset in {data_dir}. Run:  python scripts/gen_data.py")
    if not (models_dir / "distance_net.pt").exists():
        sys.exit("models/distance_net.pt missing. Run:  python scripts/train_distance_net.py")

    margin = cfg["collision"]["plan_margin"]
    d_min = np.minimum(train_data["d_body"], train_data["d_self"])
    keep = d_min > margin
    pool_q = torch.from_numpy(train_data["q"][keep])
    pool_tcp = torch.from_numpy(train_data["tcp"][keep])
    kin = ArmKinematics(cfg)
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(models_dir / "distance_net.pt", map_location="cpu"))
    dnet.eval().requires_grad_(False)                       # frozen critic of collisions
    vq, vt, _ = sample_queries(free_pool(test_data, margin), kin, N_VAL, np.random.default_rng(seed + 1),
                               c["min_target_dist"])
    vq, vt = torch.tensor(vq, dtype=torch.float32), torch.tensor(vt, dtype=torch.float32)

    tnet = TrajNet.from_config(cfg, kin.lower, kin.upper)
    opt = torch.optim.AdamW(tnet.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    n_params = sum(p.numel() for p in tnet.parameters())
    print(f"TrajNet {n_params} params | pool {len(pool_q)} free configs | {steps} steps, batch {c['batch']}, "
          f"lr {c['lr']} cosine, clip {c['grad_clip']}, {args.threads} threads")

    rep = ROOT / "reports" / "traj_net"
    rep.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        i = torch.randint(0, len(pool_q), (c["batch"],), generator=gen)
        j = torch.randint(0, len(pool_q), (c["batch"],), generator=gen)
        qs, tg = pool_q[i], pool_tcp[j]
        P = tnet(qs, tg)
        Q = curve(P, c["t_train"])
        loss, parts = level1_loss(Q, kin.tcp(P[:, -1]), tg, dnet.clearance(Q), cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tnet.parameters(), c["grad_clip"])
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1 or step % args.eval_every == 0:
            row = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()},
                   "lr": sched.get_last_lr()[0], "seconds": time.perf_counter() - t0}
            if step % args.eval_every == 0 or step == steps:
                tnet.eval()
                row.update(validate(tnet, dnet, kin, vq, vt, cfg))
                tnet.train()
                print(f"step {step:6d}  loss {row['loss']:.4f}  val {row['val_loss']:.4f}  reach med "
                      f"{row['val_reach_median_cm']:.2f} cm  pred-collide {100 * row['val_pred_collision']:.1f} %  "
                      f"({row['seconds']:.0f} s)", flush=True)
            rows.append(row)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tnet.state_dict(), models_dir / "traj_net.pt")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(rep / "progress.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    final = {k: v for k, v in rows[-1].items()}
    final.update(params=n_params, steps=steps, seed=seed, threads=args.threads, pool=len(pool_q))
    (rep / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    plot(rows, rep / "training.png")
    print(f"saved {models_dir / 'traj_net.pt'}, {rep / 'progress.csv'}, {rep / 'summary.json'}, {rep / 'training.png'}")


if __name__ == "__main__":
    main()
