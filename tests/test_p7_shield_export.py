"""Phase 7 - safety shield, deployment exports and latency."""
import time

import numpy as np
import pytest
import torch
from conftest import model_file

from spacearm.envs.space_reach_env import ACTOR_DIM, CRITIC_DIM, SpaceReachEnv
from spacearm.export import NumpyActor, export_actor_numpy, export_actor_onnx
from spacearm.models.distance_net import DistanceNet
from spacearm.rl.ppo_lag import PPOLagAgent, evaluate
from spacearm.safety.shield import SafetyShield, shield_from_config, sphere_capsule_distance

Q0 = np.array([0.3, 0.5, -0.2, -1.0, 0.4, 0.6, 0.0])


def test_sphere_capsule_distance_simple_cases():
    a = torch.tensor([[0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 0.5]])
    r = torch.tensor([0.05])
    d_above = sphere_capsule_distance(torch.tensor([0.0, 0.0, 1.0]), 0.1, a, b, r)
    d_side = sphere_capsule_distance(torch.tensor([0.3, 0.0, 0.25]), 0.1, a, b, r)
    assert d_above.item() == pytest.approx(0.35)
    assert d_side.item() == pytest.approx(0.15)


def test_analytic_obstacle_distance_matches_pybullet(sim, kin64, rng):
    for _ in range(10):
        q = rng.uniform(sim.lower * 0.9, sim.upper * 0.9)
        sim.set_q(q)
        center = sim.tcp_world() + rng.normal(0.0, 0.1, 3)
        body = sim.add_sphere(center, 0.08)
        d_pb = sim.body_distance(body, links=sim.arm_link_ids, max_dist=1.0)[0]
        a, b, r = kin64.capsules(torch.tensor(q))
        d_an = sphere_capsule_distance(torch.tensor(center), 0.08, a, b, r).min().item()
        sim.remove_body(body)
        assert d_an == pytest.approx(d_pb, abs=2e-3)


def test_shield_passes_safe_actions_through(cfg, kin64):
    shield = SafetyShield(kin64, cfg, self_clearance_fn=lambda q: torch.ones(q.shape[:-1], dtype=q.dtype))
    a = np.full(7, 0.5)
    out, info = shield.filter(Q0, a, obstacle=None)
    np.testing.assert_allclose(out, a)
    assert not info["intervened"]


def test_shield_intervenes_before_hitting_an_obstacle(cfg, kin64):
    shield = SafetyShield(kin64, cfg, self_clearance_fn=None)
    a = np.full(7, 0.8)
    step = cfg["robot"]["arm"]["max_velocity"] * cfg["shield"]["lookahead_steps"] / cfg["sim"]["control_hz"]
    q_next = torch.tensor(Q0 + a * step)
    obstacle = {"center": kin64.tcp(q_next).numpy(), "radius": 0.06}   # right where the gripper is heading
    out, info = shield.filter(Q0, a, obstacle)
    assert info["intervened"]
    assert np.all(np.abs(out) <= 1.0)
    h = lambda act: shield.safety_value(torch.tensor(Q0 + act * step), obstacle).item()  # noqa: E731
    assert h(out) > h(a)                                       # the filtered command is safer


def test_shield_with_distance_net_returns_valid_action(cfg, kin64):
    shield = shield_from_config(kin64, cfg, DistanceNet.from_config(cfg))
    out, info = shield.filter(Q0, np.full(7, 0.3))
    assert out.shape == (7,) and np.all(np.abs(out) <= 1.0) and "intervened" in info


def test_env_accepts_a_shield(test_env_cfg):
    env = SpaceReachEnv(test_env_cfg)
    env.set_shield(SafetyShield(env.kin, test_env_cfg))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros((1, 7), np.float32))
    assert "shield" in info
    env.close()


@pytest.fixture
def agent(cfg):
    torch.manual_seed(0)
    ag = PPOLagAgent(ACTOR_DIM, CRITIC_DIM, 7, cfg["ppo"])
    ag.actor_rms.update(np.random.default_rng(0).normal(1.0, 2.0, (200, ACTOR_DIM)))
    return ag


def test_numpy_actor_matches_torch(agent, tmp_path):
    actor = NumpyActor(export_actor_numpy(agent, tmp_path / "actor.npz"))
    obs = np.random.default_rng(1).normal(1.0, 2.0, (20, ACTOR_DIM)).astype(np.float32)
    np.testing.assert_allclose(actor(obs), agent.act(obs), atol=1e-5)


def test_onnx_export_matches_torch(agent, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path = export_actor_onnx(agent, tmp_path / "actor.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(2).normal(1.0, 2.0, (20, ACTOR_DIM)).astype(np.float32)
    np.testing.assert_allclose(sess.run(None, {"obs": obs})[0], agent.act(obs), atol=1e-5)


def test_numpy_actor_latency_under_one_millisecond(agent, tmp_path):
    actor = NumpyActor(export_actor_numpy(agent, tmp_path / "actor.npz"))
    obs = np.zeros(ACTOR_DIM, np.float32)
    times = []
    for _ in range(500):
        t0 = time.perf_counter()
        actor(obs)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 1e-3


class _PriorOnly:
    """Zero residual = the classical prior alone (baseline)."""

    def act(self, obs_actor):
        return np.zeros((1, 7), np.float32)


@pytest.mark.slow
@pytest.mark.needs_models
def test_trained_policy_is_safe_and_not_worse_than_the_prior(cfg):
    """Level-2 acceptance at full difficulty (S4), identical seeds for both runs:
    RL + shield must be safe (collisions <= 3 %) and at least as successful as the prior alone."""
    agent = PPOLagAgent.load(model_file(cfg, "ppo_lag.pt"))
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(model_file(cfg, "distance_net.pt"), map_location="cpu"))

    def run(policy, use_shield):
        env = SpaceReachEnv(cfg, difficulty=1.0)
        if use_shield:
            env.set_shield(shield_from_config(env.kin, cfg, dnet))
        res = evaluate(policy, env, n_episodes=40, seed=10_000)
        env.close()
        return res

    prior = run(_PriorOnly(), use_shield=False)
    ours = run(agent, use_shield=True)
    assert ours["collision"] <= 0.03
    assert ours["success"] >= prior["success"]
