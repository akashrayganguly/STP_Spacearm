"""Phase 5 - Level-2 environment (free-floating base, obstacles, faults) and vectorisation."""
import warnings

import numpy as np
import pytest
import torch

from spacearm.config import deep_update
from spacearm.control import reach_prior
from spacearm.envs.space_reach_env import ACTOR_DIM, CRITIC_DIM, SL, SpaceReachEnv, shaped_reward
from spacearm.envs.toy_env import PointHazardEnv
from spacearm.rl.vec_env import SubprocVecEnv, SyncVecEnv

INFO_KEYS = {"cost", "success", "collision", "d_tcp", "clearance"}


@pytest.fixture
def env(test_env_cfg):
    e = SpaceReachEnv(test_env_cfg)
    yield e
    e.close()


def test_spaces_and_reset(env):
    obs, info = env.reset(seed=0)
    assert obs["actor"].shape == (1, ACTOR_DIM) and obs["critic"].shape == (CRITIC_DIM,)
    assert obs["actor"].dtype == np.float32 and env.observation_space.contains(obs)
    assert np.isfinite(obs["actor"]).all() and np.isfinite(obs["critic"]).all()
    assert env.action_space.shape == (1, 7)
    assert INFO_KEYS <= set(info)


def test_passes_gymnasium_env_checker(env):
    from gymnasium.utils.env_checker import check_env

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)


def test_step_contract(env):
    env.reset(seed=1)
    obs, r, te, tr, info = env.step(np.zeros((1, 7), np.float32))
    assert isinstance(r, float) and isinstance(te, bool) and isinstance(tr, bool)
    assert INFO_KEYS <= set(info) and info["cost"] in (0.0, 1.0)
    assert env.observation_space.contains(obs)


def test_same_seed_same_episode(test_env_cfg):
    a, b = SpaceReachEnv(test_env_cfg), SpaceReachEnv(test_env_cfg)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa["actor"], ob["actor"])
    act = np.full((1, 7), 0.3, np.float32)
    for _ in range(5):
        oa = a.step(act)[0]
        ob = b.step(act)[0]
    np.testing.assert_allclose(oa["critic"], ob["critic"], atol=1e-6)
    a.close()
    b.close()


def test_truncates_at_max_steps(test_env_cfg):
    e = SpaceReachEnv(deep_update(test_env_cfg, {"env": {"max_steps": 5}}))
    e.reset(seed=3)
    for k in range(5):
        _, _, te, tr, _ = e.step(np.zeros((1, 7), np.float32))
    assert tr and not te
    e.close()


def test_spacecraft_rotates_when_the_arm_moves(test_env_cfg):
    e = SpaceReachEnv(deep_update(test_env_cfg, {"env": {"residual": False}}))
    e.reset(seed=4)
    _, orn0 = e.sim.base_pose()
    for _ in range(15):
        e.step(np.full((1, 7), 0.8, np.float32))
    _, orn1 = e.sim.base_pose()
    assert np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(orn0, orn1)))))) > 0.2
    e.close()


def test_obstacle_contact_costs_and_terminates(env):
    env.reset(seed=5)
    env.place_obstacle(env.sim.tcp_world(), 0.1)          # obstacle right on the gripper
    _, _, te, _, info = env.step(np.zeros((1, 7), np.float32))
    assert info["cost"] == 1.0 and info["collision"] and te


def test_shaped_reward_signs(cfg):
    w = cfg["env"]["reward"]
    z = np.zeros(7)
    assert shaped_reward(0.5, 0.4, z, z, False, w) > 0          # progress is rewarded
    assert shaped_reward(0.4, 0.5, z, z, False, w) < 0          # moving away is penalised
    assert shaped_reward(0.5, 0.5, np.ones(7), z, False, w) < shaped_reward(0.5, 0.5, z, z, False, w)
    assert shaped_reward(0.02, 0.01, z, z, True, w) > w["success"] * 0.9


def test_encoder_bias_is_hidden_from_actor_but_known_to_critic(test_env_cfg):
    e = SpaceReachEnv(deep_update(test_env_cfg, {"env": {"faults": {"enabled": True}}}), difficulty=1.0)
    obs, _ = e.reset(seed=6)
    q_true = e.sim.get_q()
    q_actor = e.mid + e.half * obs["actor"][0, SL["q"]]
    q_critic = e.mid + e.half * obs["critic"][ACTOR_DIM:ACTOR_DIM + 7]
    np.testing.assert_allclose(q_actor, q_true + e.bias, atol=1e-5)   # actor sees biased encoders
    np.testing.assert_allclose(q_critic, q_true, atol=1e-5)           # critic sees the truth
    assert np.abs(e.bias).max() > 0
    e.close()


def test_prior_moves_the_tcp_toward_the_target(cfg, kin64):
    q = np.array([0.3, 0.5, -0.2, -1.0, 0.4, 0.6, 0.0])
    rel = np.array([0.1, -0.05, 0.08])
    a = reach_prior(kin64, q, rel, cfg)
    assert np.all(np.abs(a) <= 1.0)
    tcp_vel = kin64.jacobian(torch.tensor(q)).numpy() @ (a * cfg["robot"]["arm"]["max_velocity"])
    assert np.dot(tcp_vel, rel) > 0


def test_zero_residual_follows_the_prior(env):
    _, info0 = env.reset(seed=8)
    d0 = info0["d_tcp"]
    for _ in range(10):
        _, _, te, tr, info = env.step(np.zeros((1, 7), np.float32))
        if te or tr:
            break
    assert info["d_tcp"] < d0


def test_targets_lie_within_the_reach_fraction(cfg, env):
    """Design change (Phase 5, option B): targets are sampled within env.target_reach_frac of the arm's reach."""
    for seed in range(20):
        env.reset(seed=seed)
        assert np.linalg.norm(env.target_w - env.shoulder) <= cfg["env"]["target_reach_frac"] * env.max_reach + 1e-9


def test_difficulty_is_clamped(env):
    env.set_difficulty(3.0)
    assert env.difficulty == 1.0
    env.set_difficulty(-1.0)
    assert env.difficulty == 0.0


def test_toy_env_contract():
    e = PointHazardEnv()
    obs, _ = e.reset(seed=0)
    assert e.observation_space.contains(obs)
    _, r, te, tr, info = e.step(np.zeros((1, 2), np.float32))
    assert isinstance(r, float) and "cost" in info


def test_sync_vec_env_autoresets_and_records_episodes():
    venv = SyncVecEnv([lambda: PointHazardEnv(max_steps=3)] * 2)
    obs = venv.reset(seed=0)
    assert obs["actor"].shape == (2, 1, 6) and obs["critic"].shape == (2, 7)
    for _ in range(3):
        obs, r, c, te, tr, infos = venv.step(np.zeros((2, 1, 2), np.float32))
    assert tr.all()
    np.testing.assert_allclose(obs["critic"][:, -1], 0.0)   # returned obs are fresh episodes (t = 0)
    eps = venv.pop_episodes()
    assert len(eps) == 2 and {"return", "cost", "length", "success", "collision"} <= set(eps[0])
    assert venv.pop_episodes() == []


def test_subprocess_vec_env_matches_sync():
    fns = [PointHazardEnv] * 2
    sync, sub = SyncVecEnv(fns), SubprocVecEnv(fns)
    try:
        np.testing.assert_array_equal(sync.reset(seed=3)["actor"], sub.reset(seed=3)["actor"])
        act = np.full((2, 1, 2), 0.5, np.float32)
        np.testing.assert_allclose(sync.step(act)[0]["actor"], sub.step(act)[0]["actor"])
    finally:
        sub.close()
