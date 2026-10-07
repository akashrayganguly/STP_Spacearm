"""Figures for docs/WRITEUP.md (v1.1): every part of Level 1 and Level 2, one consistent style.

Reads the committed results (reports/**) and the trained models; a few figures recompute small things
(the floating-base demo, a DistanceNet slice, open-loop runs, one Level-2 episode). Needs data/ for the
DistanceNet figures (python scripts/gen_data.py) and kaleido + Chrome for the 3D renders.

  reports/figures/writeup/*.png

    python scripts/make_writeup_figures.py [--only 03 12]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from spacearm.config import ROOT, load_config  # noqa: E402

OUT = ROOT / "reports" / "figures" / "writeup"
REP = ROOT / "reports"
# Validated categorical palette (slots 1-8, fixed order) and chart chrome.
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
REF, CRIT = "#898781", "#d03b3b"
METHODS = ["prior", "prior+shield", "rl", "rl+shield"]
MLABEL = {"prior": "Reflex", "prior+shield": "Reflex + shield", "rl": "RL", "rl+shield": "RL + shield"}
MCOLOR = dict(zip(METHODS, S[:4]))
STAGES = ["baseline", "network", "network+refine", "network+polish", "refine+polish", "full"]
SLABEL = {"baseline": "Baseline\nIK + line", "network": "TrajNet\nalone", "network+refine": "TrajNet\n+ refine",
          "network+polish": "TrajNet\n+ polish", "refine+polish": "+ refine\n+ polish", "full": "Full planner\n(+ verify)"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "figure.dpi": 100, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", OUT / name)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in rows[0]}


def bar_labels(ax, bars, fmt="{:.0f}", inside=False, size=8, color=INK2):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=color)


def trim(img, pad=12):
    """Crop the white border of a rendered image (H, W, C floats in [0, 1])."""
    ink = np.where(img[..., :3].min(-1) < 0.97)
    if len(ink[0]) == 0:
        return img
    y0, y1 = max(0, ink[0].min() - pad), min(img.shape[0], ink[0].max() + pad)
    x0, x1 = max(0, ink[1].min() - pad), min(img.shape[1], ink[1].max() + pad)
    return img[y0:y1, x0:x1]


def models():
    from spacearm.kinematics import ArmKinematics
    from spacearm.models.distance_net import DistanceNet

    cfg = load_config()
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(ROOT / "models" / "distance_net.pt", map_location="cpu"))
    dnet.eval().requires_grad_(False)
    return cfg, ArmKinematics(cfg, dtype=torch.float64), dnet


# ====================================================================== Level 1
def fig01_floating_base():
    """Move one joint for 2 s, then stop: the spacecraft turns the other way and stops; the CoM stays put."""
    from spacearm.sim import SpaceRobotSim

    cfg = load_config()
    sim = SpaceRobotSim(cfg)
    q0 = np.radians([0, 30, 0, 60, 0, 30, 0])
    sim.reset(q=q0)
    com0, n_sub = sim.system_com(), 24
    t, q2, rot, drift = [], [], [], []
    for k in range(41):                                   # 4 s at 10 Hz
        qd = np.zeros(7)
        if k < 20:
            qd[1] = 0.5                                   # joint 2 at its speed limit for 2 s
        sim.apply_joint_velocities(qd)
        sim.step(n_sub)
        _, orn = sim.base_pose()
        t.append((k + 1) / 10)
        q2.append(np.degrees(sim.get_q()[1] - q0[1]))
        rot.append(np.degrees(2 * np.arccos(min(1.0, abs(orn[3])))))
        drift.append(1000 * np.linalg.norm(sim.system_com() - com0))
    sim.close()
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, y, title, unit, col in ((axs[0], q2, "Arm: joint 2 turns 57°", "joint 2 change (deg)", S[0]),
                                    (axs[1], rot, "Spacecraft turns the other way", "base rotation (deg)", S[1]),
                                    (axs[2], drift, "Centre of mass does not move", "CoM drift (mm)", S[2])):
        ax.plot(t, y, color=col)
        ax.axvspan(0, 2, color=GRID, alpha=0.45, lw=0)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(unit)
        ax.set_xlim(0, 4.1)
    axs[1].annotate(f"{rot[-1]:.1f}° and holds", (t[-1], rot[-1]), xytext=(-6, -14), textcoords="offset points",
                    ha="right", fontsize=9, color=INK2)
    axs[2].set_ylim(0, 2)
    axs[2].axhline(2, color=MUTED, lw=1)
    axs[2].annotate("test limit 2 mm", (4.05, 2), xytext=(0, -12), textcoords="offset points", ha="right",
                    fontsize=9, color=MUTED)
    axs[0].annotate("arm moving", (1.0, 2), ha="center", fontsize=9, color=INK2)
    save(fig, "01_floating_base.png")


def fig02_dataset():
    from spacearm.datagen import load_dataset

    tr = load_dataset(ROOT / "data" / "kinematics_dataset.npz")
    te = load_dataset(ROOT / "data" / "kinematics_test.npz")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.1), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    bins = np.linspace(-0.25, 0.30, 56)
    for ax, key, title in ((axs[0], "d_body", "d_body: arm ↔ spacecraft"), (axs[1], "d_self", "d_self: arm ↔ arm")):
        for data, lab, col in ((te, "uniform random poses (test set)", S[0]),
                               (tr, "training set (+25 % near-surface samples)", S[1])):
            w = np.full(len(data[key]), 100.0 / len(data[key]))
            ax.hist(np.clip(data[key], bins[0], bins[-1]), bins=bins, weights=w, histtype="step", lw=2, color=col, label=lab)
        ax.axvline(0, color=CRIT, lw=1)
        ax.set_title(title)
        ax.set_xlabel("signed distance (m)  ·  negative = collision")
        ax.set_ylabel("share of samples (%)")
    axs[0].legend(loc="upper left")
    axs[0].annotate("collision", (-0.005, axs[0].get_ylim()[1] * 0.3), ha="right", fontsize=9, color=CRIT)
    axs[0].annotate("18 cm cap: link 2 always\nsits near the pedestal", (0.165, axs[0].get_ylim()[1] * 0.72),
                    ha="right", fontsize=8.5, color=INK2)
    cats = ["collide", "near surface\n(|d| < 5 cm)", "arm ↔ arm\ncollide"]
    vals_te = [100 * np.mean(np.minimum(te["d_body"], te["d_self"]) < 0),
               100 * np.mean(np.abs(np.minimum(te["d_body"], te["d_self"])) < 0.05), 100 * np.mean(te["d_self"] < 0)]
    vals_tr = [100 * np.mean(np.minimum(tr["d_body"], tr["d_self"]) < 0),
               100 * np.mean(np.abs(np.minimum(tr["d_body"], tr["d_self"])) < 0.05), 100 * np.mean(tr["d_self"] < 0)]
    x = np.arange(3)
    b1 = axs[2].bar(x - 0.19, vals_te, 0.36, color=S[0], label="uniform", edgecolor=SURF, linewidth=1.5)
    b2 = axs[2].bar(x + 0.19, vals_tr, 0.36, color=S[1], label="training", edgecolor=SURF, linewidth=1.5)
    for b in (b1, b2):
        for bar in b:
            h = bar.get_height()
            axs[2].annotate(f"{h:.1f}" if h >= 1 else f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)
    axs[2].set_xticks(x, cats, fontsize=8.5)
    axs[2].set_ylabel("% of samples")
    axs[2].set_title("What the data contains")
    axs[2].grid(axis="x", visible=False)
    save(fig, "02_dataset.png")


def fig03_distancenet():
    from spacearm.datagen import load_dataset

    cfg, kin, dnet = models()
    te = load_dataset(ROOT / "data" / "kinematics_test.npz")
    with torch.no_grad():
        q = torch.as_tensor(te["q"], dtype=torch.float32)
        pred = dnet(q).numpy()
        clear = dnet.clearance(q).numpy()
    true_min = np.minimum(te["d_body"], te["d_self"])
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1, 1.15, 1], "wspace": 0.42})
    ax = axs[0]
    ax.axvspan(-25, -5, color=GRID, alpha=0.5, lw=0)
    ax.text(-24, -22, "training targets\nclipped at −5 cm", fontsize=8, color=INK2)
    hb = ax.hexbin(100 * true_min, 100 * clear, gridsize=55, extent=(-25, 30, -25, 30), bins="log", mincnt=1,
                   cmap=LinearSegmentedColormap.from_list("b", ["#cde2fb", "#2a78d6", "#0d366b"]))
    ax.plot([-25, 30], [-25, 30], color=INK2, lw=1)
    ax.axhline(0, color=CRIT, lw=0.8)
    ax.axvline(0, color=CRIT, lw=0.8)
    ax.set_xlabel("true clearance, PyBullet (cm)")
    ax.set_ylabel("DistanceNet clearance (cm)")
    ax.set_title("Prediction vs truth")
    ax.text(-23, 26, "upper left = said safe\nbut colliding (rare)", fontsize=8.5, color=CRIT, va="top")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("poses per cell", color=INK2)
    cb.outline.set_visible(False)

    ax = axs[1]
    edges = [(-np.inf, 0, "colliding"), (0, 0.05, "0–5 cm"), (0.05, 0.10, "5–10 cm"), (0.10, 0.15, "10–15 cm"),
             (0.15, np.inf, "≥ 15 cm")]
    mae, p99, share = [], [], []
    for lo, hi, _ in edges:
        msk = (true_min >= lo) & (true_min < hi)
        err = np.abs(np.concatenate([pred[msk, 0] - np.clip(te["d_body"][msk], -0.05, 0.30),
                                     pred[msk, 1] - np.clip(te["d_self"][msk], -0.05, 0.30)]))
        mae.append(100 * err.mean())
        p99.append(100 * np.percentile(clear[msk] - np.clip(true_min[msk], -0.05, 0.30), 99))
        share.append(100 * msk.mean())
    x = np.arange(len(edges))
    b1 = ax.bar(x - 0.19, mae, 0.36, color=S[0], label="mean absolute error", edgecolor=SURF, linewidth=1.5)
    b2 = ax.bar(x + 0.19, p99, 0.36, color=S[1], label="99th-percentile over-estimate", edgecolor=SURF, linewidth=1.5)
    bar_labels(ax, b1, "{:.1f}")
    bar_labels(ax, b2, "{:.1f}")
    ax.set_xticks(x, [f"{e[2]}\n{s:.0f} %" for e, s in zip(edges, share)], fontsize=8.5)
    ax.set_xlabel("true clearance band · share of test poses")
    ax.set_ylabel("cm")
    ax.set_title("Error by true clearance band")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)

    ax = axs[2]
    pg = read_csv(REP / "distance_net" / "progress.csv")
    ax.plot(pg["epoch"], pg["test_mae_cm"], color=S[0])
    ax.axhline(1.0, color=MUTED, lw=1)
    ax.annotate("target ≤ 1 cm", (60, 1.0), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    ax.annotate(f"{pg['test_mae_cm'][-1]:.2f} cm", (pg["epoch"][-1], pg["test_mae_cm"][-1]), xytext=(-4, -14),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("epoch (60 epochs took 49 s on 4 CPU cores)")
    ax.set_ylabel("test mean absolute error (cm)")
    ax.set_title("Training")
    save(fig, "03_distancenet.png")


def fig04_distance_slice():
    """The clearance 'map' over two joints (others fixed): PyBullet truth vs DistanceNet."""
    from spacearm.sim import SpaceRobotSim

    cfg, kin, dnet = models()
    sim = SpaceRobotSim(cfg)
    base = np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0])
    n = 81
    a2 = np.radians(np.linspace(-120, 120, n))
    a4 = np.radians(np.linspace(-120, 120, n))
    Q = np.repeat(base[None], n * n, 0)
    Q[:, 1] = np.repeat(a2, n)
    Q[:, 3] = np.tile(a4, n)
    truth = np.empty(len(Q))
    for k, q in enumerate(Q):
        sim.set_q(q)
        truth[k] = min(sim.min_distances())
    sim.close()
    with torch.no_grad():
        pred = dnet.clearance(torch.as_tensor(Q, dtype=torch.float32)).numpy()
    T, P = 100 * truth.reshape(n, n), 100 * pred.reshape(n, n)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    ext = [-120, 120, -120, 120]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, Z, title in ((axs[0], T, "Truth (PyBullet)"), (axs[1], P, "DistanceNet prediction")):
        im = ax.imshow(Z.T, origin="lower", extent=ext, cmap=cmap, vmin=-25, vmax=25, aspect="auto")
        ax.contour(np.degrees(a2), np.degrees(a4), Z.T, levels=[0], colors=[INK], linewidths=1.6)
        ax.contour(np.degrees(a2), np.degrees(a4), Z.T, levels=[5], colors=[INK2], linewidths=0.9, linestyles="solid")
        ax.set_title(title)
        ax.set_xlabel("joint 2 (deg)")
        ax.grid(False)
    axs[0].set_ylabel("joint 4 (deg)")
    cb = fig.colorbar(im, ax=axs[:2], pad=0.02, shrink=0.9)
    cb.set_label("clearance (cm): red = collision, blue = free", color=INK2)
    cb.outline.set_visible(False)
    E = np.where(T >= -5, P - T, np.nan)                  # below -5 cm the training targets were clipped
    axs[2].set_facecolor("#e8e7e2")
    im = axs[2].imshow(E.T, origin="lower", extent=ext, cmap=cmap.reversed(), vmin=-5, vmax=5, aspect="auto")
    axs[2].contour(np.degrees(a2), np.degrees(a4), T.T, levels=[0], colors=[INK], linewidths=1.2)
    axs[2].set_title("Error = DistanceNet − truth")
    axs[2].set_xlabel("joint 2 (deg)")
    axs[2].grid(False)
    cb = fig.colorbar(im, ax=axs[2], pad=0.02)
    cb.set_label("cm (red = over-estimates clearance)", color=INK2)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.06, "Joints 1, 3, 5, 7 = 0°, joint 6 = 30°. Black line: contact (0 cm); thin line: 5 cm planning margin. "
             f"Error shown where the truth is above −5 cm (grey: deeper collisions, whose training targets were clipped); "
             f"mean |error| there {np.nanmean(np.abs(E)):.2f} cm.", fontsize=9, color=INK2)
    save(fig, "04_distance_slice.png")


def fig05_bezier():
    st = json.loads((REP / "level1" / "stages.json").read_text())
    ex = st["example"]["stages"]["refine+polish"]
    cp = np.degrees(np.array(ex["control_points"]))
    path = np.degrees(np.array(ex["path"]))
    s = np.linspace(0, 1, 201)
    W = np.stack([comb(7, k) * s ** k * (1 - s) ** (7 - k) for k in range(8)], 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axs[0]
    for k in range(8):
        ax.plot(s, W[:, k], color=S[k], lw=2, label=f"P{k}")
    frames = np.linspace(0, 1, 16)
    ax.plot(frames, np.full(16, -0.08), "|", color=INK2, ms=9, mew=1.5)
    ax.annotate("ticks: the 16 frames scored in training and refining", (0.5, -0.08), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("fraction of the path s")
    ax.set_ylabel("weight of control point k")
    ax.set_title("Each pose = weighted average of the 8 control points")
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.2, columnspacing=0.8)
    ax = axs[1]
    dev = np.abs(cp[2:6] - (cp[0] + (cp[-1] - cp[0]) * (np.arange(2, 6) / 7)[:, None])).max(0)
    j = int(np.argmax(dev))
    straight = cp[0, j] + (cp[-1, j] - cp[0, j]) * np.arange(8) / 7
    ax.plot(np.linspace(0, 1, len(path)), path[:, j], color=S[0], lw=2.5, label="the path (what the arm does)")
    ax.scatter(np.arange(8) / 7, cp[:, j], s=60, color=S[1], edgecolor=SURF, linewidth=2, zorder=3,
               label="control points (start ×2, 4 middle, goal ×2)")
    ax.scatter(np.arange(2, 6) / 7, straight[2:6], s=40, facecolor=SURF, edgecolor=MUTED, linewidth=1.5, zorder=2,
               label="middle points' straight-line spots")
    for k in range(2, 6):
        ax.annotate("", (k / 7, cp[k, j]), (k / 7, straight[k]),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
    ax.set_xlabel("fraction of the path s")
    ax.set_ylabel(f"joint {j + 1} angle (deg)")
    ax.set_title("One joint of a real plan: pulled toward, not through, the middle points")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "05_bezier.png")


def fig06_trajnet_training():
    pg = read_csv(REP / "traj_net" / "progress.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    ax = axs[0]
    for key, w, col, lab in (("reach", 100, S[0], "100 × reach² (tip to target)"),
                             ("coll", 1000, S[1], "1000 × clearance shortfall²"),
                             ("smooth", 0.1, S[2], "0.1 × smoothness")):
        ax.plot(pg["step"], w * pg[key], color=col, lw=1.6, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("training step (256 queries each)")
    ax.set_ylabel("loss term (log scale)")
    ax.set_title("The three parts of the score while TrajNet learns")
    ax.legend(loc="upper right")
    ax = axs[1]
    msk = ~np.isnan(pg["val_reach_lt_2cm"])
    ax.plot(pg["step"][msk], 100 * pg["val_reach_lt_2cm"][msk], color=S[0], marker="o", ms=5, mec=SURF, mew=1.5,
            label="tip within 2 cm")
    ax.plot(pg["step"][msk], 100 * pg["val_pred_collision"][msk], color=S[1], marker="o", ms=5, mec=SURF, mew=1.5,
            label="path predicted to collide")
    ax.set_ylim(0, 100)
    ax.set_xlabel("training step")
    ax.set_ylabel("% of 512 fixed validation queries")
    ax.set_title("Network alone on unseen queries")
    ax.legend(loc="center right")
    ax.annotate(f"{100 * pg['val_reach_lt_2cm'][msk][-1]:.0f} %", (pg["step"][msk][-1], 100 * pg["val_reach_lt_2cm"][msk][-1]),
                xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    save(fig, "06_trajnet_training.png")


def fig07_planner_stages():
    st = json.loads((REP / "level1" / "stages.json").read_text())
    x = np.arange(len(STAGES))
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 6.6))
    panels = [(axs[0, 0], "success", 100, "verified success (%)", "Success: no contact, tip < 2 cm, within limits", "{:.0f}"),
              (axs[0, 1], "collision_rate", 100, "paths that touch (%)", "Collisions (PyBullet, 100 points per path)", "{:.1f}"),
              (axs[1, 0], "median_reach_err_cm", 1, "median reach error (cm)", "How close the tip gets to the target", "{:.2f}"),
              (axs[1, 1], "median_time_ms", 1, "median time (ms, 1 CPU thread)", "Planning time", "{:.0f}")]
    for ax, key, mult, ylab, title, fmt in panels:
        for k, (set_name, col) in enumerate((("random", S[0]), ("hard", S[1]))):
            vals = [mult * st[set_name][s][key] for s in STAGES]
            bars = ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, edgecolor=SURF, linewidth=1.5,
                          label=f"{set_name} queries ({st[set_name]['full']['n']})")
            bar_labels(ax, bars, fmt, size=7.5)
        ax.set_xticks(x, [SLABEL[s] for s in STAGES], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axs[0, 0].set_ylim(0, 112)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=10)
    axs[1, 0].set_ylim(0, 2.4)
    axs[1, 0].axhline(2.0, color=CRIT, lw=1)
    axs[1, 0].annotate("2 cm success limit", (5.4, 2.0), xytext=(0, 3), textcoords="offset points", ha="right",
                       fontsize=8.5, color=CRIT)
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_ylim(0.5, 600)
    fig.text(0.01, -0.01, "Hard queries: the classical baseline collides. Times: median, 1 CPU thread on this cloud VM, "
             "which ran 1.7× slower than the v1.0 machine (full planner there: 99 ms, or 114 ms with the check).",
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "07_planner_stages.png")


def fig08_worked_example():
    st = json.loads((REP / "level1" / "stages.json").read_text())
    ex = st["example"]
    fig = plt.figure(figsize=(14, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    names = [("baseline", REF, "Baseline: IK + straight line"), ("network", S[0], "TrajNet alone"),
             ("network+refine", S[1], "+ refine"), ("refine+polish", S[2], "+ refine + IK polish")]
    for key, col, lab in names:
        c = np.array(ex["stages"][key]["clearance_cm"])
        ax.plot(np.linspace(0, 1, len(c)), c, color=col, lw=2,
                label=f"{lab}: min {c.min():+.1f} cm, tip {ex['stages'][key]['reach_err_cm']:.1f} cm off")
    ax.axhline(0, color=CRIT, lw=1)
    ax.axhline(5, color=MUTED, lw=1)
    ax.annotate("contact", (0.45, 0), xytext=(0, 3), textcoords="offset points", fontsize=9, color=CRIT)
    ax.annotate("5 cm planning margin", (0.45, 5), xytext=(0, 3), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xlabel("fraction of the path s")
    ax.set_ylabel("true clearance (cm)")
    ax.set_title("A hard query, stage by stage")
    ax.legend(loc="lower left", fontsize=8.3)
    for k, (fname, title) in enumerate((("08a_baseline_3d.png", "Baseline (red): sweeps through the payload"),
                                        ("08b_planner_3d.png", "Planner: arcs over it (tip paths)"))):
        a = fig.add_subplot(gs[k + 1])
        if (OUT / fname).exists():
            a.imshow(trim(plt.imread(OUT / fname)))
        a.axis("off")
        a.set_title(title, fontsize=10, loc="center")
    save(fig, "08_worked_example.png")


def render_3d_example():
    """Two 3D renders of the worked example (kaleido + Chrome)."""
    import plotly.graph_objects as go

    from spacearm.gui import scene

    st = json.loads((REP / "level1" / "stages.json").read_text())
    ex = st["example"]
    cfg, kin, _ = models()
    cam = {"eye": {"x": 0.75, "y": -0.55, "z": 0.42}, "center": {"x": -0.02, "y": 0.0, "z": 0.12}}

    def tip(Q):
        with torch.no_grad():
            return kin.forward(torch.as_tensor(np.asarray(Q)))[1][:, -1].numpy()

    base = np.array(ex["stages"]["baseline"]["path"])
    worst = base[int(np.argmin(ex["stages"]["baseline"]["clearance_cm"]))]
    final = np.array(ex["stages"]["refine+polish"]["path"])
    target = scene.point_trace(ex["target"], scene.COLORS["target"], "target", size=10)
    renders = {
        "08a_baseline_3d.png": [scene.spacecraft_trace(cfg), scene.arm_trace(kin, ex["q_start"], opacity=0.35, ghost=True),
                                scene.arm_trace(kin, worst, color="#e34948"), scene.line_trace(tip(base), REF, "tip", 7),
                                target],
        "08b_planner_3d.png": [scene.spacecraft_trace(cfg), scene.arm_trace(kin, ex["q_start"], opacity=0.35, ghost=True),
                               scene.arm_trace(kin, final[45], opacity=0.25, ghost=True), scene.arm_trace(kin, final[-1]),
                               scene.line_trace(tip(np.array(ex["stages"]["network"]["path"])), S[0], "TrajNet", 7),
                               scene.line_trace(tip(final), S[2], "final", 7), target]}
    for name, traces in renders.items():
        fig = go.Figure(traces, layout=scene.layout(height=520, show_axes=False, legend=False))
        fig.update_layout(scene_camera=cam, paper_bgcolor="white", scene_bgcolor="white")
        fig.write_image(OUT / name, width=640, height=520, scale=1.5)
        print("saved", OUT / name)


def fig09_openloop():
    """A perfect body-frame plan, played open-loop on the floating spacecraft: the tip misses the target."""
    from spacearm.datagen import load_dataset
    from spacearm.kinematics import ArmKinematics
    from spacearm.models.traj_net import TrajNet
    from spacearm.planner import Level1Planner, free_pool, plan_and_verify, sample_queries
    from spacearm.sim import SpaceRobotSim

    cfg, kin64, dnet = models()
    kin = ArmKinematics(cfg)
    tnet = TrajNet.from_config(cfg, kin.lower, kin.upper)
    tnet.load_state_dict(torch.load(ROOT / "models" / "traj_net.pt", map_location="cpu"))
    planner = Level1Planner(kin, dnet, tnet, cfg)
    sim = SpaceRobotSim(cfg)
    pool = free_pool(load_dataset(ROOT / "data" / "kinematics_test.npz"), 0.05)
    qs, ts, _ = sample_queries(pool, kin, 40, np.random.default_rng(10), 0.20)
    runs = []
    hz, v = cfg["sim"]["control_hz"], cfg["robot"]["arm"]["max_velocity"]
    n_sub = int(round(cfg["sim"]["physics_hz"] / hz))
    for q, t in zip(qs, ts):
        Q, info = plan_and_verify(planner, sim, q, t)
        if not info["success"]:
            continue
        sim.reset(q=Q[0])
        seg = np.abs(np.diff(Q, axis=0)).max(1) / v
        knots = np.concatenate([[0.0], np.cumsum(seg)])
        dur = knots[-1]
        rec = {"t": [], "miss": [], "body": [], "rot": []}
        for k in range(1, int(np.ceil(dur * hz)) + 2 * hz + 1):
            tt = min(k / hz, dur)
            qref = np.array([np.interp(tt, knots, Q[:, j]) for j in range(7)])
            sim.apply_joint_velocities((qref - sim.get_q()) * hz)
            sim.step(n_sub)
            _, orn = sim.base_pose()
            rec["t"].append(k / hz)
            rec["miss"].append(100 * np.linalg.norm(sim.tcp_world() - t))
            rec["body"].append(100 * np.linalg.norm(sim.tcp_body() - t))
            rec["rot"].append(np.degrees(2 * np.arccos(min(1.0, abs(orn[3])))))
        runs.append({k: np.array(val) for k, val in rec.items()})
        if len(runs) == 20:
            break
    sim.close()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3), gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    ex = max(runs, key=lambda r: r["miss"][-1] if r["miss"][-1] < 25 else 0)
    ax = axs[0]
    ax.plot(ex["t"], ex["body"], color=S[0], label="as planned (spacecraft frame)")
    ax.plot(ex["t"], ex["miss"], color=S[1], label="in space (where the target really is)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tip to target (cm)")
    ax.set_title("One plan, played without feedback")
    ax.legend(loc="upper right")
    ax.annotate(f"{ex['miss'][-1]:.0f} cm miss", (ex["t"][-1], ex["miss"][-1]), xytext=(-4, 6), textcoords="offset points",
                ha="right", fontsize=9, color=INK2)
    ax = axs[1]
    ax.plot(ex["t"], ex["rot"], color=S[2])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spacecraft rotation (deg)")
    ax.set_title("…because the spacecraft turned")
    ax = axs[2]
    miss = np.array([r["miss"][-1] for r in runs])
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(-0.15, 0.15, len(miss)), miss, s=36, color=S[1], edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.hlines(np.median(miss), -0.3, 0.3, color=INK, lw=2)
    ax.annotate(f"median {np.median(miss):.0f} cm", (0.32, np.median(miss)), va="center", fontsize=9, color=INK2)
    ax.set_xlim(-0.5, 1.0)
    ax.set_xticks([])
    ax.set_ylabel("final miss in space (cm)")
    ax.set_title(f"{len(runs)} verified plans")
    ax.grid(axis="x", visible=False)
    save(fig, "09_openloop.png")
    (OUT / "09_openloop.json").write_text(json.dumps({"n": len(runs), "median_miss_cm": float(np.median(miss)),
                                                      "max_miss_cm": float(miss.max()),
                                                      "median_rotation_deg": float(np.median([r["rot"][-1] for r in runs]))}))


# ====================================================================== Level 2
def fig10_episode():
    """The demo episode (S2): the reflex stalls in front of the ball; RL + shield goes around it."""
    from spacearm.gui import backend as B

    res = json.loads((REP / "level2" / "eval_S2.json").read_text())
    prior = {e["seed"]: e for e in res["episodes_detail"]["prior"]}
    rl = {e["seed"]: e for e in res["episodes_detail"]["rl+shield"]}
    good = [s for s in prior if prior[s]["obstacle"] and not prior[s]["success"] and not prior[s]["collision"]
            and rl[s]["success"] and rl[s]["obstacle"] and rl[s]["length"] >= 60]
    seed = min(good, key=lambda s: rl[s]["length"])
    spec = B.MissionSpec(seed=seed, source="S2 obstacles only")
    tr = {c: B.run_mission(c, spec, render=False) for c in ("Reflex only", "RL + shield")}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [0.001, 1, 1]})
    axs[0].axis("off")
    cols = {"Reflex only": S[0], "RL + shield": S[3]}
    a = tr["RL + shield"]
    ax = axs[1]
    for c, t in tr.items():
        ax.plot(t["t"], 100 * t["d_tcp"], color=cols[c], label=f"{c}: {t['outcome'].lower()}")
    ax.axvline(a["sampled"]["obstacle_s"], color=MUTED, lw=1)
    ax.annotate("ball appears", (a["sampled"]["obstacle_s"], ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tip to target (cm)")
    ax.set_title("Distance to the target")
    ax.legend(loc="upper right", fontsize=8.5)
    ax = axs[2]
    for c, t in tr.items():
        d = 100 * np.where(t["d_obs"] < 0.299, t["d_obs"], np.nan)
        ax.plot(t["t"], d, color=cols[c], label=c)
    on = a["t"][a["shield"] > 0]
    if len(on):
        ax.plot(on, np.full(len(on), 0.6), "|", color=CRIT, ms=8, label=f"shield intervened ({len(on)} steps)")
    ax.set_ylim(0, None)
    ax.axhline(2, color=MUTED, lw=1)
    ax.annotate("2 cm near-miss line", (a["t"][-1], 2), xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=9, color=MUTED)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("clearance to the ball (cm)")
    ax.set_title("Keeping clear of the ball (arm to ball, true)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "10_episode.png")


def scene_green():
    return "#0ca30c"


def fig11_ppo_training():
    pg = read_csv(REP / "ppo_lag" / "progress.csv")
    ev = read_csv(REP / "ppo_lag" / "eval.csv")
    k = 25
    sm = lambda y: np.convolve(y, np.ones(k) / k, mode="valid")  # noqa: E731
    m = (pg["steps"] / 1e6)[k // 2: k // 2 + len(pg["steps"]) - k + 1]
    mfull = pg["steps"] / 1e6
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    ax = axs[0]
    ax.plot(m, 100 * sm(pg["success"]), color=S[0], lw=1.6, label="training episodes (current difficulty)")
    ax.plot(ev["steps"] / 1e6, 100 * ev["success"], color=S[1], marker="o", ms=4, mec=SURF, mew=1, lw=1.4,
            label="test at full difficulty (20 episodes)")
    ax.axhline(100 * ev["prior_success"][0], color=REF, lw=1)
    ax.annotate("reflex alone", (3.0, 100 * ev["prior_success"][0]), xytext=(0, -12), textcoords="offset points",
                ha="right", fontsize=8.5, color=REF)
    ax.set_ylim(0, 100)
    ax.set_title("Success (%)")
    ax.legend(loc="lower left", fontsize=8)
    ax = axs[1]
    ax.plot(m, sm(pg["cost"]), color=S[0], lw=1.6)
    ax.axhline(1.0, color=CRIT, lw=1)
    ax.annotate("limit: 1 near-miss step / episode", (3.0, 1.0), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=CRIT)
    ax.set_ylim(0, 1.6)
    ax.set_title("Near-miss cost per episode")
    ax = axs[2]
    ax.plot(mfull, pg["lambda"], color=S[2], lw=1.2)
    ax.set_title("Safety weight λ (learned)")
    ax.set_ylim(0, 1)
    ax = axs[3]
    ax.plot(mfull, pg["difficulty"], color=S[6], lw=2)
    ax.set_title("Curriculum: difficulty d")
    ax.set_ylim(0, 1.05)
    for ax in axs:
        ax.set_xlabel("environment steps (millions)")
    save(fig, "11_ppo_training.png")


def _scenario_results():
    out = {}
    for s in ("S1", "S2", "S3", "S4"):
        out[s] = json.loads((REP / "level2" / f"eval_{s}.json").read_text())
    return out


def fig12_scenarios():
    res = _scenario_results()
    names = {"S1": "S1 nominal\n(no surprises)", "S2": "S2 obstacles\nonly", "S3": "S3 faults +\nnoise only",
             "S4": "S4 everything\n(d = 1)"}
    x = np.arange(4)
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, key, title in ((axs[0], "success", "Success (%)"), (axs[1], "collision", "Collisions (%)")):
        for k, m in enumerate(METHODS):
            vals = [100 * res[s]["methods"][m][key] for s in res]
            bars = ax.bar(x + (k - 1.5) * 0.19, vals, 0.19, color=MCOLOR[m], edgecolor=SURF, linewidth=1.5, label=MLABEL[m])
            bar_labels(ax, bars, "{:.0f}", size=7.5)
        ax.set_xticks(x, [names[s] for s in res], fontsize=9)
        ax.set_title(title + " · 100 identical episodes per bar")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylim(0, 105)
    axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.1, -0.18))
    for s, xi in (("S2", 1), ("S4", 3)):
        v = 100 * res[s]["methods"]["prior (reflex off)"]["collision"]
        axs[1].annotate(f"reflex off:\n{v:.0f} %", (xi - 0.29, 8.5), ha="center", fontsize=8, color=REF)
    axs[1].set_ylim(0, 10)
    save(fig, "12_scenarios.png")


def _variants():
    d = REP / "level2" / "variants"
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def fig13_difficulty():
    v = _variants()
    keys = [k for k in ("d0.00", "d0.25", "d0.50", "d0.75", "d1.00") if k in v]
    ds = [v[k]["difficulty"] for k in keys]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, key, mult, title, ylab in ((axs[0], "success", 100, "Success", "%"),
                                       (axs[1], "collision", 100, "Collisions", "%"),
                                       (axs[2], "time_to_reach_s", 1, "Time to reach (successful episodes)", "s")):
        for m in METHODS:
            y = [mult * v[k]["methods"][m][key] for k in keys]
            ax.plot(ds, y, color=MCOLOR[m], marker="o", ms=6, mec=SURF, mew=2, label=MLABEL[m])
            ax.annotate(f"{y[-1]:.0f}" if mult == 100 else f"{y[-1]:.1f}", (ds[-1], y[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_xlabel("difficulty d (surprises scaled together)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.12)
    axs[0].set_ylim(0, 100)
    axs[1].set_ylim(-0.3, 8)
    axs[0].legend(loc="lower left", fontsize=8.5)
    save(fig, "13_difficulty.png")


VARIANT_ORDER = ["obs_static", "obs_drift", "obs_small", "obs_large", "bias", "weak", "slip", "noise",
                 "stress_xl_ball", "stress_fast_ball", "stress_bias4", "stress_noise3"]
GROUP_LABEL = {"obstacle": "Obstacle only", "faults": "Faults only", "noise": "Noise only", "stress": "Stress tests"}


def fig14_variants():
    v = _variants()
    keys = [k for k in VARIANT_ORDER if k in v]
    y = np.arange(len(keys))[::-1].astype(float)
    # gaps between groups
    prev = None
    for i, k in enumerate(keys):
        if prev is not None and v[k]["group"] != prev:
            y[i:] -= 0.8
        prev = v[k]["group"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.42 * len(keys) + 1.6), sharey=True,
                            gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, key, title, lim in ((axs[0], "success", "Success (%)", (0, 100)),
                                (axs[1], "collision", "Collisions (%)", (-0.5, 12))):
        for i, k in enumerate(keys):
            vals = [100 * v[k]["methods"][m][key] for m in METHODS]
            ax.hlines(y[i], min(vals), max(vals), color=GRID, lw=10, zorder=1, alpha=0.6)
            for j, (m, val) in enumerate(zip(METHODS, vals)):
                ax.scatter(val, y[i] + (1.5 - j) * 0.13, s=55, color=MCOLOR[m], edgecolor=SURF, linewidth=1.5, zorder=3,
                           label=MLABEL[m] if i == 0 else None)
        ax.set_xlim(*lim)
        ax.set_title(title + " · 100 identical episodes per dot")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [v[k]["label"] for k in keys], fontsize=9)
    prev = None
    for i, k in enumerate(keys):
        if v[k]["group"] != prev:
            axs[0].annotate(GROUP_LABEL[v[k]["group"]], (0, y[i] + 0.55), xytext=(-8, 0), textcoords="offset points",
                            xycoords=("axes fraction", "data"), ha="right", fontsize=9, fontweight="bold", color=INK)
            prev = v[k]["group"]
    axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.8, -0.08), fontsize=9)
    save(fig, "14_variants.png")


def fig15_shield_margin():
    sw = json.loads((REP / "level2" / "margin_sweep_s30000_n100.json").read_text())
    acc = json.loads((REP / "level2" / "margin_sweep_s10000_n40.json").read_text())
    ms = sorted(sw["margins"], key=float)
    x = [100 * float(m) for m in ms]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, key, title in ((axs[0], "success", "Success (%) · 100 episodes"),
                           (axs[2], "shield_intervention_rate", "Shield intervenes (% of steps)")):
        for m in ("prior+shield", "rl+shield"):
            yv = [100 * sw["margins"][k][m][key] for k in ms]
            ax.plot(x, yv, color=MCOLOR[m], marker="o", ms=6, mec=SURF, mew=2, label=MLABEL[m])
        ax.axvline(6, color=MUTED, lw=1)
        ax.set_xlabel("shield self-margin (cm)")
        ax.set_title(title)
        ax.set_xticks(x)
    axs[0].legend(loc="lower left")
    ax = axs[1]
    am = sorted(acc["margins"], key=float)
    xx = np.arange(len(am))
    for j, m in enumerate(("prior+shield", "rl+shield")):
        vals = [round(acc["margins"][k][m]["collision"] * acc["episodes"]) for k in am]
        bars = ax.bar(xx + (j - 0.5) * 0.34, vals, 0.34, color=MCOLOR[m], edgecolor=SURF, linewidth=1.5, label=MLABEL[m])
        bar_labels(ax, bars, "{:.0f}")
    ax.set_xticks(xx, [f"{100 * float(k):.0f} cm" for k in am])
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("episodes with a collision")
    ax.set_title(f"Collisions in the {acc['episodes']} acceptance episodes")
    ax.grid(axis="x", visible=False)
    ax.annotate("100 other episodes (seeds 30000+):\n0 collisions at every margin", (0.5, 2.55), ha="center",
                fontsize=8.5, color=INK2)
    fig.text(0.01, -0.06, "S4 (everything on). The margin is how much predicted clearance the shield insists on; "
             "6 cm was chosen (grey line): it removes the acceptance-test grazes for 2 points of success.",
             fontsize=9, color=INK2)
    save(fig, "15_shield_margin.png")


def fig16_latency():
    lat = json.loads((REP / "level2" / "latency.json").read_text())["latency"]
    rows = [("actor_onnx", "Policy (ONNX Runtime)"), ("actor_numpy", "Policy (NumPy)"), ("actor_torch", "Policy (PyTorch)"),
            ("shield_pass", "Shield: command accepted"), ("prior", "Reflex prior"),
            ("shield_obstacle", "Shield: escape with obstacle"), ("env_step_with_shield", "Whole control step\n(incl. physics)")]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(rows))[::-1]
    p50 = [lat[k]["p50_ms"] for k, _ in rows]
    p99 = [lat[k]["p99_ms"] for k, _ in rows]
    ax.hlines(y, p50, p99, color=AXIS, lw=3)
    ax.scatter(p50, y, s=60, color=S[0], edgecolor=SURF, linewidth=2, zorder=3, label="median")
    ax.scatter(p99, y, s=40, color=S[1], edgecolor=SURF, linewidth=2, zorder=3, label="99th percentile")
    for yi, a in zip(y, p50):
        ax.annotate(f"{a:.3f} ms" if a < 0.1 else f"{a:.2f} ms", (a, yi), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.axvline(100, color=CRIT, lw=1)
    ax.annotate("control period\n100 ms", (100, y[0]), xytext=(-5, 0), textcoords="offset points", ha="right",
                va="center", fontsize=9, color=CRIT)
    ax.set_xscale("log")
    ax.set_xlim(0.004, 200)
    ax.set_yticks(y, [r[1] for r in rows], fontsize=9)
    ax.set_xlabel("milliseconds on one CPU thread (log scale)")
    ax.set_title("Latency: everything fits in the 100 ms control period with room to spare")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.6, 0.62))
    save(fig, "16_latency.png")


FIGS = {"01": fig01_floating_base, "02": fig02_dataset, "03": fig03_distancenet, "04": fig04_distance_slice,
        "05": fig05_bezier, "06": fig06_trajnet_training, "07": fig07_planner_stages, "08r": render_3d_example,
        "08": fig08_worked_example, "09": fig09_openloop, "10": fig10_episode, "11": fig11_ppo_training,
        "12": fig12_scenarios, "13": fig13_difficulty, "14": fig14_variants, "15": fig15_shield_margin,
        "16": fig16_latency}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, choices=list(FIGS))
    args = ap.parse_args()
    torch.set_num_threads(1)
    if "BROWSER_PATH" not in os.environ and Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome").exists():
        os.environ["BROWSER_PATH"] = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"   # cloud VM only
    for k in args.only or list(FIGS):
        try:
            FIGS[k]()
        except Exception as e:                       # one broken figure should not stop the others
            print(f"figure {k} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
