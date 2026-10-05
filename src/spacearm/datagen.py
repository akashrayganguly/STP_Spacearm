"""Synthetic kinematics/collision dataset (DESIGN §5.1), generated in PyBullet.

Each sample: joint angles `q` (7), signed distances `d_body` and `d_self` (m, negative = penetration,
clipped at sim.max_query_dist) and the TCP position `tcp` (3) in the body frame. All float32.

A share of the samples (`data.boundary_frac`) is drawn near the collision surface: near-surface
seeds (|min(d_body, d_self)| < boundary_band) are perturbed with N(0, boundary_sigma) and only
perturbations that stay near the surface are kept. That is where the DistanceNet must be accurate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KEYS = ("q", "d_body", "d_self", "tcp")
_MAX_BOUNDARY_ROUNDS = 50          # safety cap on rejection rounds (each round labels one batch)


def sample_configs(n: int, lower, upper, rng: np.random.Generator, frac: float = 1.0) -> np.ndarray:
    """n joint configurations drawn uniformly inside `frac` of each joint range (centred), shape (n, 7)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    mid, half = (lower + upper) / 2, frac * (upper - lower) / 2
    return rng.uniform(mid - half, mid + half, (int(n), len(lower)))


def label_configs(sim, Q) -> dict[str, np.ndarray]:
    """Label configurations with PyBullet: signed body/self distances and the body-frame TCP position."""
    Q = np.asarray(Q, float).reshape(-1, len(sim.arm_joint_ids))
    d = np.empty((len(Q), 2))
    tcp = np.empty((len(Q), 3))
    for i, q in enumerate(Q):
        sim.set_q(q)
        d[i] = sim.min_distances()
        tcp[i] = sim.tcp_body()
    return {"q": Q.astype(np.float32), "d_body": d[:, 0].astype(np.float32),
            "d_self": d[:, 1].astype(np.float32), "tcp": tcp.astype(np.float32)}


def _concat(parts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def _take(data: dict, idx) -> dict[str, np.ndarray]:
    return {k: data[k][idx] for k in KEYS}


def _near(data: dict, band: float) -> np.ndarray:
    return np.abs(np.minimum(data["d_body"], data["d_self"])) < band


def generate_distance_dataset(sim, n: int, rng: np.random.Generator, cfg: dict) -> dict[str, np.ndarray]:
    """n labelled samples: (1 - boundary_frac) uniform + boundary_frac near the collision surface, shuffled.

    Deterministic for a given `rng` state (PyBullet distance queries are deterministic)."""
    dc = cfg["data"]
    frac, band, sigma = dc["limit_frac"], dc["boundary_band"], dc["boundary_sigma"]
    n_bnd = int(round(dc["boundary_frac"] * n))
    n_uni = int(n) - n_bnd
    mid, half = (sim.lower + sim.upper) / 2, frac * (sim.upper - sim.lower) / 2
    lo, hi = mid - half, mid + half

    uniform = label_configs(sim, sample_configs(n_uni, sim.lower, sim.upper, rng, frac))
    parts = [uniform]
    if n_bnd > 0:
        # Seeds: near-surface uniform samples (draw extra uniform samples just for seeding if there are none).
        seeds = uniform["q"][_near(uniform, band)]
        while len(seeds) == 0:
            extra = label_configs(sim, sample_configs(max(n_bnd, 100), sim.lower, sim.upper, rng, frac))
            seeds = extra["q"][_near(extra, band)]
        found, n_found, last = [], 0, None
        for _ in range(_MAX_BOUNDARY_ROUNDS):
            batch = n_bnd - n_found
            base = seeds[rng.integers(0, len(seeds), batch)].astype(float)
            cand = label_configs(sim, np.clip(base + rng.normal(0.0, sigma, base.shape), lo, hi))
            keep = _near(cand, band)
            found.append(_take(cand, keep))
            n_found += int(keep.sum())
            last = _take(cand, ~keep)
            if n_found >= n_bnd:
                break
        if n_found < n_bnd:                       # cap reached (very unlikely): top up with the last rejects
            found.append(_take(last, slice(0, n_bnd - n_found)))
        parts.append(_take(_concat(found), slice(0, n_bnd)))

    data = _concat(parts)
    return _take(data, rng.permutation(len(data["q"])))


def save_dataset(path: str | Path, data: dict) -> Path:
    """Save as an (uncompressed) .npz; returns the path actually written."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(data[k]) for k in KEYS})
    return path


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load a dataset written by `save_dataset` into memory."""
    with np.load(Path(path)) as f:
        return {k: f[k] for k in KEYS}
