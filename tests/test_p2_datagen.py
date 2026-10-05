"""Phase 2 - synthetic kinematics/collision dataset."""
import numpy as np
import torch

from spacearm.config import deep_update
from spacearm.datagen import generate_distance_dataset, label_configs, load_dataset, sample_configs, save_dataset

KEYS = {"q", "d_body", "d_self", "tcp"}


def test_sample_configs_within_limits(sim, rng):
    Q = sample_configs(1000, sim.lower, sim.upper, rng, frac=0.98)
    assert Q.shape == (1000, 7)
    mid, half = (sim.lower + sim.upper) / 2, (sim.upper - sim.lower) / 2
    assert np.all(np.abs(Q - mid) <= 0.98 * half + 1e-12)


def test_labels_are_consistent_with_kinematics(cfg, sim, kin64, rng):
    Q = sample_configs(50, sim.lower, sim.upper, rng)
    data = label_configs(sim, Q)
    assert set(data) == KEYS
    assert all(v.dtype == np.float32 for v in data.values())
    tcp_fk = kin64.tcp(torch.tensor(data["q"], dtype=torch.float64)).numpy()
    np.testing.assert_allclose(data["tcp"], tcp_fk, atol=1e-5)
    assert np.all(data["d_body"] <= cfg["sim"]["max_query_dist"] + 1e-6)
    assert np.all(data["d_self"] <= cfg["sim"]["max_query_dist"] + 1e-6)


def test_dataset_shapes_and_collision_share(cfg, sim):
    data = generate_distance_dataset(sim, 400, np.random.default_rng(1), cfg)
    assert set(data) == KEYS and len(data["q"]) == 400
    assert data["tcp"].shape == (400, 3)
    collide = np.minimum(data["d_body"], data["d_self"]) < 0
    assert 0.02 < collide.mean() < 0.6        # both classes present, neither dominates


def test_dataset_is_reproducible(cfg, sim):
    a = generate_distance_dataset(sim, 200, np.random.default_rng(7), cfg)
    b = generate_distance_dataset(sim, 200, np.random.default_rng(7), cfg)
    for k in KEYS:
        np.testing.assert_array_equal(a[k], b[k])


def test_boundary_sampling_enriches_the_collision_surface(cfg, sim):
    band = cfg["data"]["boundary_band"]
    uni = generate_distance_dataset(sim, 600, np.random.default_rng(3),
                                    deep_update(cfg, {"data": {"boundary_frac": 0.0}}))
    bnd = generate_distance_dataset(sim, 600, np.random.default_rng(3),
                                    deep_update(cfg, {"data": {"boundary_frac": 0.5}}))
    near = lambda d: np.mean(np.abs(np.minimum(d["d_body"], d["d_self"])) < band)  # noqa: E731
    assert near(bnd) > 1.5 * near(uni)


def test_save_load_roundtrip(tmp_path, cfg, sim):
    data = generate_distance_dataset(sim, 50, np.random.default_rng(0), cfg)
    path = save_dataset(tmp_path / "d.npz", data)
    back = load_dataset(path)
    for k in KEYS:
        np.testing.assert_array_equal(back[k], data[k])
