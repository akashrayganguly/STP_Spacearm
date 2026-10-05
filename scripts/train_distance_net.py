"""Train the DistanceNet (DESIGN §5.2) on data/kinematics_dataset.npz; report on data/kinematics_test.npz.

  models/distance_net.pt                 trained weights (state_dict)
  reports/distance_net/metrics.json      test-set metrics vs DESIGN §8 (+ validation, per head, near-surface)
  reports/distance_net/progress.csv      per-epoch losses and test MAE
  reports/distance_net/training.png      loss curves, test MAE, predicted vs true clearance

    python scripts/train_distance_net.py [--config ...] [--seed 0] [--epochs 60] [--threads 4]
(run python scripts/gen_data.py first if data/ is empty)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from spacearm.models.distance_net import DistanceNet, distance_loss  # noqa: E402

VAL_FRAC = 0.05               # held-out share of the training file, for monitoring only
NEAR_EVAL = 0.10              # m, "|d| < 10 cm" MAE band of DESIGN §8
FALSE_SAFE_AT = 0.03          # m, predicted clearance above this while actually colliding = "false safe"
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, CRITICAL = "#2a78d6", "#eb6834", "#c62828"


def to_tensors(data: dict, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.from_numpy(data["q"])
    y = torch.from_numpy(np.stack([data["d_body"], data["d_self"]], 1)).clamp(*cfg["distance_net"]["clip"])
    return q, y


@torch.no_grad()
def predict(net: DistanceNet, q: torch.Tensor, batch: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    preds, clears = [], []
    for i in range(0, len(q), batch):
        d = net(q[i:i + batch])
        preds.append(d)
        clears.append(-net.tau * torch.logsumexp(-d / net.tau, dim=-1))
    return torch.cat(preds), torch.cat(clears)


@torch.no_grad()
def metrics(net: DistanceNet, q: torch.Tensor, y: torch.Tensor, cfg: dict) -> dict:
    """Same definitions as the Phase 3 acceptance test (clipped targets, clearance = smooth min)."""
    pred, clear = predict(net, q)
    err = (pred - y).abs()
    true_min = y.min(1).values
    near = y.abs() < NEAR_EVAL
    band = cfg["distance_net"]["near_band"]
    out = {
        "n": len(q),
        "mae_all_cm": err.mean().item() * 100,
        "mae_near10_cm": err[near].mean().item() * 100,
        "mae_body_cm": err[:, 0].mean().item() * 100,
        "mae_self_cm": err[:, 1].mean().item() * 100,
        "sign_accuracy": ((clear > 0) == (true_min > 0)).float().mean().item(),
        "false_safe_rate": ((clear > FALSE_SAFE_AT) & (true_min < 0)).float().mean().item(),
        "false_safe_any_rate": ((clear > 0) & (true_min < 0)).float().mean().item(),
        "collision_share": (true_min < 0).float().mean().item(),
        "max_overestimate_cm": (clear - true_min).max().item() * 100,
    }
    self_near = y[:, 1].abs() < band            # the thin arm-arm surface (Phase 2 open issue)
    out["self_near_surface_n"] = int(self_near.sum())
    out["self_near_surface_mae_cm"] = err[self_near, 1].mean().item() * 100 if self_near.any() else float("nan")
    return out


def plot(rows: list[dict], net: DistanceNet, q_test: torch.Tensor, y_test: torch.Tensor, path: Path) -> None:
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_MUTED)
        ax.grid(color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)
    a = axes[0]
    a.plot(ep, [r["train_loss"] for r in rows], color=BLUE, lw=2, label="train")
    a.plot(ep, [r["val_loss"] for r in rows], color=ORANGE, lw=2, label="validation (5 % held out)")
    a.set_yscale("log")
    a.set_title("Weighted Huber loss", color=INK)
    a.set_xlabel("epoch", color=INK_MUTED)
    a.legend(frameon=False, labelcolor=INK)
    a = axes[1]
    a.plot(ep, [r["test_mae_cm"] for r in rows], color=BLUE, lw=2)
    a.axhline(1.0, color=INK_MUTED, ls="--", lw=1)
    a.text(ep[-1], 1.0, "target 1.0 cm", color=INK_MUTED, ha="right", va="bottom", fontsize=9)
    a.set_ylim(0, max(2.5, min(10.0, max(r["test_mae_cm"] for r in rows) * 1.05)))
    a.set_title("Test MAE (uniform 20k, both heads)", color=INK)
    a.set_xlabel("epoch", color=INK_MUTED)
    a.set_ylabel("cm", color=INK_MUTED)
    a = axes[2]
    _, clear = predict(net, q_test)
    t = y_test.min(1).values.numpy() * 100
    c = clear.numpy() * 100
    a.axvspan(-6, 0, ymin=0, ymax=1, color=GRID, alpha=0.5, lw=0)
    a.fill_between([-6, 0], FALSE_SAFE_AT * 100, 32, color=CRITICAL, alpha=0.08, lw=0)
    a.text(-5.5, 30, "false safe", color=CRITICAL, fontsize=9, va="top")
    a.scatter(t, c, s=4, color=BLUE, alpha=0.25, lw=0)
    a.plot([-6, 32], [-6, 32], color=INK_MUTED, lw=1, ls="--")
    a.set_xlim(-6, 32)
    a.set_ylim(-6, 32)
    a.set_title("Clearance on the test set", color=INK)
    a.set_xlabel("true min distance (cm, clipped)", color=INK_MUTED)
    a.set_ylabel("predicted clearance (cm)", color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="default: distance_net.epochs")
    ap.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg["distance_net"]
    seed = cfg["seed"] if args.seed is None else args.seed
    epochs = c["epochs"] if args.epochs is None else args.epochs
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    data_dir = ROOT / cfg["paths"]["data_dir"]
    try:
        train_np, test_np = load_dataset(data_dir / "kinematics_dataset.npz"), load_dataset(data_dir / "kinematics_test.npz")
    except FileNotFoundError:
        sys.exit(f"No dataset in {data_dir}. Run:  python scripts/gen_data.py")
    q_all, y_all = to_tensors(train_np, cfg)
    q_test, y_test = to_tensors(test_np, cfg)
    perm = torch.randperm(len(q_all), generator=gen)
    n_val = int(round(VAL_FRAC * len(q_all)))
    q_val, y_val = q_all[perm[:n_val]], y_all[perm[:n_val]]
    q_tr, y_tr = q_all[perm[n_val:]], y_all[perm[n_val:]]

    net = DistanceNet.from_config(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    steps_per_epoch = math.ceil(len(q_tr) / c["batch"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=0.0)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DistanceNet {n_params} params | train {len(q_tr)} val {len(q_val)} test {len(q_test)} | "
          f"{epochs} epochs x {steps_per_epoch} steps, batch {c['batch']}, lr {c['lr']} cosine, {args.threads} threads")

    rep = ROOT / "reports" / "distance_net"
    rep.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(q_tr), generator=gen)
        total = 0.0
        for i in range(0, len(q_tr), c["batch"]):
            idx = order[i:i + c["batch"]]
            loss = distance_loss(net(q_tr[idx]), y_tr[idx], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_loss = distance_loss(predict(net, q_val)[0], y_val, cfg).item()
            test_mae = (predict(net, q_test)[0] - y_test).abs().mean().item() * 100
        rows.append({"epoch": epoch, "train_loss": total / len(q_tr), "val_loss": val_loss, "test_mae_cm": test_mae,
                     "lr": sched.get_last_lr()[0], "seconds": time.perf_counter() - t0})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            r = rows[-1]
            print(f"epoch {epoch:3d}  train {r['train_loss']:.3e}  val {r['val_loss']:.3e}  "
                  f"test MAE {test_mae:.2f} cm  ({r['seconds']:.0f} s)")

    models_dir = ROOT / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), models_dir / "distance_net.pt")

    net.eval()
    test_m, val_m = metrics(net, q_test, y_test, cfg), metrics(net, q_val, y_val, cfg)
    targets = {"mae_all_cm": ("<=", 1.0), "mae_near10_cm": ("<=", 2.0), "sign_accuracy": (">=", 0.97),
               "false_safe_rate": ("<=", 0.005)}
    checks = {k: {"target": f"{op} {v}", "got": test_m[k], "pass": test_m[k] <= v if op == "<=" else test_m[k] >= v}
              for k, (op, v) in targets.items()}
    report = {"test_uniform": test_m, "validation_train_distribution": val_m, "targets": checks,
              "params": n_params, "epochs": epochs, "seed": seed, "train_seconds": rows[-1]["seconds"],
              "threads": args.threads}
    (rep / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(rep / "progress.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, net, q_test, y_test, rep / "training.png")

    print(f"\nTest set (uniform, n={test_m['n']}):")
    for k, ch in checks.items():
        print(f"  {k:18s} {ch['got']:.4f}  (target {ch['target']})  {'PASS' if ch['pass'] else 'FAIL'}")
    print(f"  per head MAE: body {test_m['mae_body_cm']:.2f} cm, self {test_m['mae_self_cm']:.2f} cm; "
          f"self near-surface MAE {test_m['self_near_surface_mae_cm']:.2f} cm (n={test_m['self_near_surface_n']})")
    print(f"saved {models_dir / 'distance_net.pt'}, {rep / 'metrics.json'}, {rep / 'progress.csv'}, "
          f"{rep / 'training.png'}")


if __name__ == "__main__":
    main()
