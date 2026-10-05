"""Phase 4 - Level-1 trajectory model, losses and planner."""
import time

import numpy as np
import pytest
import torch
from conftest import model_file

from spacearm.datagen import generate_distance_dataset
from spacearm.kinematics import ArmKinematics
from spacearm.losses import collision_loss, level1_loss, limit_violation, reach_loss, smoothness_loss
from spacearm.models.distance_net import DistanceNet
from spacearm.models.traj_net import TrajNet, bezier_basis, curve, make_control_points, squash, unsquash
from spacearm.planner import (Level1Planner, free_pool, ik_baseline, polish_ik, sample_queries,
                              verify_trajectory)


@pytest.fixture
def kin32(cfg):
    return ArmKinematics(cfg)


def test_bezier_basis_is_a_partition_of_unity():
    B = bezier_basis(50, degree=7)
    assert B.shape == (50, 8)
    torch.testing.assert_close(B.sum(-1), torch.ones(50))
    assert torch.all(B >= 0)
    assert B[0, 0] == 1 and B[-1, -1] == 1


def test_squash_stays_inside_limits_and_inverts(kin64):
    z = torch.tensor([[-1e6] * 7, [0.0] * 7, [1e6] * 7], dtype=torch.float64)
    q = squash(z, kin64.lower, kin64.upper)
    assert torch.all(q >= kin64.lower) and torch.all(q <= kin64.upper)
    q2 = torch.rand(10, 7, dtype=torch.float64) * 0.9 * (kin64.upper - kin64.lower) + kin64.lower + 0.05
    torch.testing.assert_close(squash(unsquash(q2, kin64.lower, kin64.upper), kin64.lower, kin64.upper), q2)


def test_control_points_pin_start_and_goal(cfg, kin32):
    torch.manual_seed(0)
    net = TrajNet.from_config(cfg, kin32.lower, kin32.upper)
    for p in net.parameters():                            # random (untrained) outputs
        torch.nn.init.normal_(p, std=0.5)
    q_start = (torch.rand(6, 7) - 0.5) * (kin32.upper - kin32.lower) * 0.9
    P = net(q_start, torch.randn(6, 3))
    assert P.shape == (6, cfg["traj"]["degree"] + 1, 7)
    torch.testing.assert_close(P[:, 0], q_start)
    torch.testing.assert_close(P[:, 1], q_start)
    torch.testing.assert_close(P[:, -1], P[:, -2])


def test_whole_curve_respects_joint_limits_for_extreme_outputs(kin32):
    q_start = torch.zeros(4, 7)
    u_goal = torch.randn(4, 7) * 100
    delta = torch.randn(4, 4, 7) * 100
    Q = curve(make_control_points(q_start, u_goal, delta, kin32.lower, kin32.upper), 200)
    assert limit_violation(Q, kin32.lower, kin32.upper) == 0      # convex-hull property


def test_rest_to_rest_motion(kin32):
    P = make_control_points(torch.zeros(1, 7), torch.ones(1, 7), torch.randn(1, 4, 7), kin32.lower, kin32.upper)
    Q = curve(P, 1001)[0]
    v_mid = (Q[501] - Q[500]).abs().max()
    # P0 = P1 and P6 = P7 make dQ/ds = 0 at both ends, so the first step is O(h^2), not O(h)
    assert (Q[1] - Q[0]).abs().max() < 0.02 * v_mid
    assert (Q[-1] - Q[-2]).abs().max() < 0.02 * v_mid


def test_losses_behave():
    t = torch.randn(5, 3)
    assert reach_loss(t, t) == 0 and reach_loss(t, t + 0.1) > 0
    assert collision_loss(torch.full((5,), 0.2), 0.05) == 0
    assert collision_loss(torch.full((5,), 0.0), 0.05) > 0
    assert smoothness_loss(torch.ones(2, 10, 7)) == 0


def test_level1_loss_backpropagates_into_the_network(cfg, kin32):
    torch.manual_seed(0)
    net = TrajNet.from_config(cfg, kin32.lower, kin32.upper)
    dnet = DistanceNet.from_config(cfg)
    q_start, target = torch.zeros(8, 7), torch.full((8, 3), 0.5)
    P = net(q_start, target)
    Q = curve(P, cfg["traj"]["t_train"])
    loss, parts = level1_loss(Q, kin32.tcp(P[:, -1]), target, dnet.clearance(Q), cfg)
    loss.backward()
    assert set(parts) == {"reach", "coll", "smooth"}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_polish_ik_reaches_mm_precision(kin64, rng):
    for _ in range(5):
        q_goal = torch.tensor(rng.uniform(-1.5, 1.5, 7))
        target = kin64.tcp(q_goal)
        q0 = q_goal + torch.tensor(rng.normal(0, 0.05, 7))
        q = polish_ik(kin64, q0, target, iters=10)
        assert torch.linalg.norm(kin64.tcp(q) - target) < 1e-3


def test_baseline_and_verifier_contract(cfg, sim, kin64):
    data = generate_distance_dataset(sim, 300, np.random.default_rng(0), cfg)
    pool = free_pool(data, cfg["collision"]["plan_margin"])
    qs, tg, qg = sample_queries(pool, kin64, 3, np.random.default_rng(1), cfg["traj"]["min_target_dist"])
    assert qs.shape == (3, 7) and tg.shape == (3, 3) and qg.shape == (3, 7)
    Q = ik_baseline(sim, qs[0], tg[0], 20)
    assert Q.shape == (20, 7)
    np.testing.assert_allclose(Q[0], qs[0], atol=1e-6)
    res = verify_trajectory(sim, Q, tg[0], cfg["traj"]["success_tol"])
    assert {"collision_free", "min_dist", "reach_err", "within_limits", "success"} <= set(res)


@pytest.mark.needs_models
def test_trained_planner_meets_targets(cfg, sim, kin32):
    """Acceptance check for Phase 4 (random queries): success >= 90 %, < 0.5 s per plan."""
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(model_file(cfg, "distance_net.pt"), map_location="cpu"))
    tnet = TrajNet.from_config(cfg, kin32.lower, kin32.upper)
    tnet.load_state_dict(torch.load(model_file(cfg, "traj_net.pt"), map_location="cpu"))
    planner = Level1Planner(kin32, dnet, tnet, cfg)
    data = generate_distance_dataset(sim, 2000, np.random.default_rng(5), cfg)
    pool = free_pool(data, cfg["collision"]["plan_margin"])
    qs, tg, _ = sample_queries(pool, kin32, 30, np.random.default_rng(6), cfg["traj"]["min_target_dist"])
    ok, times = [], []
    for k in range(len(qs)):
        t0 = time.perf_counter()
        Q, _ = planner.plan(qs[k], tg[k])
        times.append(time.perf_counter() - t0)
        ok.append(verify_trajectory(sim, Q, tg[k], cfg["traj"]["success_tol"])["success"])
    assert np.mean(ok) >= 0.9
    assert np.median(times) < 0.5
