"""Generate the Level-1 kinematics/collision datasets (DESIGN §5.1).

  data/kinematics_dataset.npz  data.n_samples (200k): uniform + boundary samples   (git-ignored)
  data/kinematics_test.npz     20k uniform samples from a different seed           (git-ignored)
  reports/phase2/dataset_stats.json, reports/figures/phase2_distance_hist.png      (committed)

Same seed -> same data, so data/ can be regenerated any time.

    python scripts/gen_data.py [--config ...] [--seed 0] [--n 200000] [--n-test 20000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spacearm.config import ROOT, deep_update, load_config  # noqa: E402
from spacearm.datagen import generate_distance_dataset, save_dataset  # noqa: E402
from spacearm.sim import SpaceRobotSim  # noqa: E402

TEST_SEED_OFFSET = 1          # test set seed = seed + 1 (a different random stream)
# Chart styling (validated categorical slots 1-2 on the light surface; text in neutral ink).
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"train": "#2a78d6", "test": "#eb6834"}


def stats(data: dict, band: float) -> dict:
    d_min = np.minimum(data["d_body"], data["d_self"])
    pct = lambda x: {f"p{k}": float(np.percentile(x, k)) for k in (1, 5, 50, 95, 99)}  # noqa: E731
    return {
        "n": int(len(d_min)),
        "collision_share": float(np.mean(d_min < 0)),
        "body_collision_share": float(np.mean(data["d_body"] < 0)),
        "self_collision_share": float(np.mean(data["d_self"] < 0)),
        "near_surface_share": float(np.mean(np.abs(d_min) < band)),
        "d_body": pct(data["d_body"]), "d_self": pct(data["d_self"]),
        "tcp_min": data["tcp"].min(0).tolist(), "tcp_max": data["tcp"].max(0).tolist(),
    }


def plot(train: dict, test: dict, band: float, clip: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    lo = min(float(np.minimum(d["d_body"], d["d_self"]).min()) for d in (train, test))
    bins = np.arange(np.floor(lo / 0.01) * 0.01, clip + 0.02, 0.01)       # 1 cm bins covering every label
    for ax, key, title in zip(axes, ("d_body", "d_self"), ("arm vs spacecraft (d_body)", "arm vs arm (d_self)")):
        ax.set_facecolor(SURFACE)
        ax.axvspan(-band, band, color=GRID, alpha=0.6, lw=0, label=f"near surface (|d| < {band * 100:.0f} cm)")
        ax.axvline(0.0, color=INK_MUTED, lw=1, ls="--")
        for name, data in (("train", train), ("test", test)):
            ax.hist(data[key], bins=bins, density=True, histtype="step", lw=2, color=SERIES[name],
                    label=f"{name} ({len(data[key]) // 1000}k{', uniform' if name == 'test' else ''})")
        ax.set_yscale("log")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("signed distance (m)  —  negative = penetration", color=INK_MUTED)
        ax.tick_params(colors=INK_MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("density (log scale)", color=INK_MUTED)
    axes[1].legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    fig.suptitle(f"Phase 2 dataset: signed-distance labels (clipped at {clip * 100:.0f} cm)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="default: seed from the config")
    ap.add_argument("--n", type=int, default=None, help="training samples (default: data.n_samples)")
    ap.add_argument("--n-test", type=int, default=20000, help="uniform test samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    n = cfg["data"]["n_samples"] if args.n is None else args.n
    band, clip = cfg["data"]["boundary_band"], cfg["sim"]["max_query_dist"]
    data_dir = ROOT / cfg["paths"]["data_dir"]
    rep_dir, fig_dir = ROOT / "reports" / "phase2", ROOT / "reports" / "figures"
    for d in (data_dir, rep_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    sim = SpaceRobotSim(cfg)
    try:
        t0 = time.perf_counter()
        train = generate_distance_dataset(sim, n, np.random.default_rng(seed), cfg)
        t_train = time.perf_counter() - t0
        p_train = save_dataset(data_dir / "kinematics_dataset.npz", train)
        print(f"train: {n} samples in {t_train:.1f} s ({t_train / n * 1e3:.3f} ms/sample) -> {p_train}")

        t0 = time.perf_counter()
        test_cfg = deep_update(cfg, {"data": {"boundary_frac": 0.0}})
        test = generate_distance_dataset(sim, args.n_test, np.random.default_rng(seed + TEST_SEED_OFFSET), test_cfg)
        t_test = time.perf_counter() - t0
        p_test = save_dataset(data_dir / "kinematics_test.npz", test)
        print(f"test:  {args.n_test} uniform samples in {t_test:.1f} s -> {p_test}")
    finally:
        sim.close()

    report = {"seed": seed, "test_seed": seed + TEST_SEED_OFFSET, "boundary_frac": cfg["data"]["boundary_frac"],
              "train": stats(train, band), "test": stats(test, band),
              "seconds": {"train": t_train, "test": t_test}, "ms_per_sample_train": t_train / n * 1e3}
    (rep_dir / "dataset_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(train, test, band, clip, fig_dir / "phase2_distance_hist.png")
    for name in ("train", "test"):
        s = report[name]
        print(f"{name:5s}: collide {s['collision_share'] * 100:5.1f} % (body {s['body_collision_share'] * 100:.1f} %, "
              f"self {s['self_collision_share'] * 100:.2f} %), near surface {s['near_surface_share'] * 100:.1f} %")
    print(f"saved {rep_dir / 'dataset_stats.json'} and {fig_dir / 'phase2_distance_hist.png'}")


if __name__ == "__main__":
    main()
