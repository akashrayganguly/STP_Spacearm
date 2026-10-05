"""Phase 3 - neural collision-distance field (Level 1a)."""
import math

import numpy as np
import pytest
import torch
from conftest import model_file

from spacearm.datagen import generate_distance_dataset, label_configs, sample_configs
from spacearm.models.distance_net import DistanceNet, distance_loss


@pytest.fixture
def net(cfg):
    torch.manual_seed(0)
    return DistanceNet.from_config(cfg)


def test_shapes(net):
    q = torch.randn(5, 7)
    assert net(q).shape == (5, 2)
    assert net.clearance(q).shape == (5,)
    assert DistanceNet.features(q).shape == (5, 14)


def test_periodic_in_joint_angles(net):
    q = torch.randn(8, 7)
    torch.testing.assert_close(net(q), net(q + 2 * math.pi), atol=1e-5, rtol=0)


def test_clearance_is_a_conservative_smooth_min(net):
    q = torch.randn(64, 7)
    d, c = net(q), net.clearance(q)
    hard_min = d.min(-1).values
    assert torch.all(c <= hard_min + 1e-6)
    assert torch.all(c >= hard_min - net.tau * math.log(2) - 1e-6)


def test_gradient_wrt_joints_is_finite_and_nonzero(net):
    q = torch.randn(4, 7, requires_grad=True)
    net.clearance(q).sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_parameter_budget(net):
    assert sum(p.numel() for p in net.parameters()) < 300_000


def test_loss_weights_errors_near_the_surface_more(cfg):
    pred = torch.zeros(1, 2)
    near = distance_loss(pred, torch.full((1, 2), 0.01), cfg)    # true distance 1 cm
    far = distance_loss(pred + 0.2, torch.full((1, 2), 0.21), cfg)  # same 1 cm error, far away
    assert near > far


def test_can_fit_a_small_labelled_batch(cfg, sim):
    torch.manual_seed(0)
    data = label_configs(sim, sample_configs(256, sim.lower, sim.upper, np.random.default_rng(0)))
    q = torch.tensor(data["q"])
    y = torch.tensor(np.stack([data["d_body"], data["d_self"]], 1)).clamp(*cfg["distance_net"]["clip"])
    net = DistanceNet.from_config(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(500):
        loss = distance_loss(net(q), y, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (net(q) - y).abs().mean().item() < 0.01           # < 1 cm on the training batch


@pytest.mark.needs_models
def test_trained_model_meets_targets(cfg, sim):
    """Acceptance check for the Phase-3 model on fresh uniform samples."""
    net = DistanceNet.from_config(cfg)
    net.load_state_dict(torch.load(model_file(cfg, "distance_net.pt"), map_location="cpu"))
    net.eval()
    data = generate_distance_dataset(sim, 2000, np.random.default_rng(12345),
                                     {**cfg, "data": {**cfg["data"], "boundary_frac": 0.0}})
    y = torch.tensor(np.stack([data["d_body"], data["d_self"]], 1)).clamp(*cfg["distance_net"]["clip"])
    with torch.no_grad():
        pred = net(torch.tensor(data["q"]))
        clear = net.clearance(torch.tensor(data["q"]))
    true_min = y.min(1).values
    assert (pred - y).abs().mean() < 0.010                     # MAE < 1 cm
    assert ((clear > 0) == (true_min > 0)).float().mean() > 0.97  # collision sign accuracy
    assert ((clear > 0.03) & (true_min < 0)).float().mean() < 0.005  # "false safe" rate
