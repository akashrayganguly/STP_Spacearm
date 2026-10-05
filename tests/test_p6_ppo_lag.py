"""Phase 6 - PPO-Lagrangian learner (checked on the toy task before the expensive arm task)."""
import math

import numpy as np
import pytest
import torch

from spacearm.config import deep_update
from spacearm.envs.toy_env import PointHazardEnv
from spacearm.rl.ppo_lag import (Actor, LagrangeMultiplier, PPOLagAgent, RunningMeanStd, compute_gae, evaluate,
                                 train)


def _col(x):
    return np.asarray(x, dtype=np.float64)[:, None]      # (T,) -> (T, N=1)


def test_gae_without_episode_end():
    r, v = _col([1, 1, 1]), _col([0, 0, 0])
    nv, z = _col([0, 0, 10]), _col([0, 0, 0])
    adv, ret = compute_gae(r, v, nv, z, z, gamma=0.9, lam=1.0)
    np.testing.assert_allclose(adv[:, 0], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(ret, adv + v)


def test_gae_termination_stops_bootstrap_and_trace():
    r, v, nv = _col([1, 1, 1]), _col([0, 0, 0]), _col([0, 0, 10])
    done = _col([0, 1, 0])
    adv, _ = compute_gae(r, v, nv, done, done, gamma=0.9, lam=1.0)
    np.testing.assert_allclose(adv[:, 0], [1.9, 1.0, 10.0])


def test_gae_truncation_bootstraps_but_cuts_trace():
    r, v, nv = _col([1, 1, 1]), _col([0, 0, 0]), _col([0, 4, 10])
    adv, _ = compute_gae(r, v, nv, _col([0, 0, 0]), _col([0, 1, 0]), gamma=0.9, lam=1.0)
    np.testing.assert_allclose(adv[:, 0], [5.14, 4.6, 10.0])


def test_lagrange_multiplier_dual_ascent():
    lam = LagrangeMultiplier(init=0.1, lr=0.5, max_value=2.0)
    assert lam.update(mean_episode_cost=3.0, cost_limit=1.0) == pytest.approx(1.1)   # violated -> up
    assert lam.update(mean_episode_cost=0.0, cost_limit=1.0) == pytest.approx(0.6)   # satisfied -> down
    assert lam.update(mean_episode_cost=0.0, cost_limit=1.0) == pytest.approx(0.1)
    assert lam.update(mean_episode_cost=0.0, cost_limit=1.0) == 0.0                  # never negative
    for _ in range(10):
        lam.update(100.0, 1.0)
    assert lam.value == 2.0                                                          # capped


def test_running_mean_std_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, (1000, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.split(x, 10):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, x.mean(0), atol=1e-3)
    np.testing.assert_allclose(rms.var, x.var(0), rtol=1e-2)


def test_actor_is_small_and_bounded(cfg):
    p = cfg["ppo"]
    actor = Actor(56, 7, p["actor_hidden"], p["log_std_init"])
    assert sum(t.numel() for t in actor.parameters()) < 20_000
    with torch.no_grad():
        a = actor(torch.randn(100, 56) * 100)
    assert torch.all(a.abs() <= 1.0)
    assert actor.log_std.exp().mean().item() == pytest.approx(math.exp(p["log_std_init"]), rel=1e-5)


def test_agent_save_and_load_roundtrip(cfg, tmp_path):
    agent = PPOLagAgent(6, 7, 2, cfg["ppo"])
    agent.actor_rms.update(np.random.default_rng(0).normal(size=(50, 6)))
    agent.save(tmp_path / "a.pt")
    back = PPOLagAgent.load(tmp_path / "a.pt")
    o = np.random.default_rng(1).normal(size=(5, 6)).astype(np.float32)
    np.testing.assert_allclose(agent.act(o), back.act(o), atol=1e-7)


def test_training_smoke_on_toy_task(cfg):
    small = deep_update(cfg, {"ppo": {"n_envs": 2, "n_steps": 64, "minibatches": 2, "epochs": 2}})
    agent, hist = train(PointHazardEnv, small, total_steps=256, seed=0, log_fn=None)
    assert len(hist) == 2
    assert {"steps", "lambda", "return", "cost", "success"} <= set(hist[-1])
    assert all(np.isfinite(t.detach().numpy()).all() for t in agent.actor.parameters())


@pytest.mark.slow
def test_ppo_lagrangian_learns_to_respect_the_constraint(cfg):
    """An unconstrained learner cuts through the hazard (about 7 cost per episode). The Lagrangian
    learner must reach the goal AND keep the expected cost near the limit (1.0)."""
    # the toy task has no prior: explore more and do not hold the actor back
    toy = deep_update(cfg, {"ppo": {"n_envs": 8, "n_steps": 128, "log_std_init": -0.5, "actor_lr": 3e-4,
                                    "actor_warmup_updates": 0}})
    agent, hist = train(PointHazardEnv, toy, total_steps=100_000, seed=0, log_fn=None)
    res = evaluate(agent, PointHazardEnv(), n_episodes=50)
    assert res["success"] >= 0.8
    assert res["cost"] <= 1.5
    assert max(h["lambda"] for h in hist) > 0.2              # the constraint was actually active
