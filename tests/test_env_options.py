"""v1.1 - SpaceReachEnv.reset(options=...): pin the start, target and surprises (used by the GUI and the
variant evaluation). Without options every episode must stay exactly as before."""
import numpy as np
import pytest

from spacearm.config import deep_update
from spacearm.envs.space_reach_env import SL, SpaceReachEnv


@pytest.fixture
def env(test_env_cfg):
    e = SpaceReachEnv(test_env_cfg, difficulty=1.0)
    yield e
    e.close()


def _free_start(env, seed):
    """A collision-free start and a target from a normal reset (so the test does not hard-code poses)."""
    env.reset(seed=seed)
    return env.sim.get_q().copy(), env.target_w.copy()


def test_empty_options_change_nothing(test_env_cfg):
    cfg = deep_update(test_env_cfg, {"env": {"obstacle": {"prob": 1.0}, "faults": {"enabled": True},
                                             "noise": {"enabled": True}}})
    a, b = SpaceReachEnv(cfg, difficulty=1.0), SpaceReachEnv(cfg, difficulty=1.0)
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7, options={})
    np.testing.assert_array_equal(oa["critic"], ob["critic"])
    act = np.full((1, 7), 0.2, np.float32)
    for _ in range(30):
        oa = a.step(act)[0]
        ob = b.step(act)[0]
    np.testing.assert_array_equal(oa["critic"], ob["critic"])
    a.close()
    b.close()


def test_pinned_start_and_target(env):
    q0, target = _free_start(env, 3)
    _, other_target = _free_start(env, 4)
    env.reset(seed=11, options={"q_start": q0, "target": other_target})
    np.testing.assert_allclose(env.sim.get_q(), q0, atol=1e-9)
    np.testing.assert_allclose(env.target_w, other_target)
    assert abs(env.d_tcp - np.linalg.norm(env.sim.tcp_world() - other_target)) < 1e-12


def test_pinned_obstacle_appears_on_time_where_asked(env):
    q0, target = _free_start(env, 5)
    center = env.sim.tcp_world() + np.array([0.0, 0.0, 0.40])
    env.reset(seed=5, options={"q_start": q0, "target": target,
                               "obstacle": {"time": 0.5, "radius": 0.07, "velocity": [0.0, 0.02, 0.0], "center": center}})
    for _ in range(4):
        env.step(np.zeros((1, 7), np.float32))
        assert env.obstacle is None
    env.step(np.zeros((1, 7), np.float32))                       # t = 5 steps = 0.5 s
    assert env.obstacle is not None and env.obstacle["radius"] == pytest.approx(0.07)
    np.testing.assert_allclose(env.obstacle["center"], center)
    env.step(np.zeros((1, 7), np.float32))
    np.testing.assert_allclose(env.obstacle["center"], center + [0.0, 0.02 * env.dt, 0.0], atol=1e-12)


def test_no_obstacle_option_overrides_the_scenario(test_env_cfg):
    cfg = deep_update(test_env_cfg, {"env": {"obstacle": {"prob": 1.0}}})
    e = SpaceReachEnv(cfg, difficulty=1.0)
    e.reset(seed=2, options={"obstacle": None})
    for _ in range(60):
        e.step(np.zeros((1, 7), np.float32))
    assert e.obstacle is None
    e.close()


def test_pinned_faults_and_noise(env):
    bias = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -2.0])
    obs, _ = env.reset(seed=8, options={"faults": {"bias_deg": bias, "gains": 0.8, "slip_time": 0.3,
                                                   "slip_joint": 2, "slip_deg": -3.0},
                                        "noise": {"encoder_deg": 0.0, "vision_m": 0.0, "gyro": 0.0}})
    np.testing.assert_allclose(env.gains, 0.8)
    q_meas = obs["actor"][0, SL["q"]] * env.half + env.mid
    np.testing.assert_allclose(q_meas - env.sim.get_q(), np.radians(bias), atol=1e-5)
    for _ in range(3):
        obs = env.step(np.zeros((1, 7), np.float32))[0]
    expected = np.radians(bias) + np.radians([0, 0, -3.0, 0, 0, 0, 0])
    q_meas = obs["actor"][0, SL["q"]] * env.half + env.mid
    np.testing.assert_allclose(q_meas - env.sim.get_q(), expected, atol=1e-5)
    env.reset(seed=8, options={"noise": {"encoder_deg": 0.1, "vision_m": 0.005, "gyro": 0.002}})
    assert env.sig_enc == pytest.approx(np.radians(0.1)) and env.sig_vis == 0.005 and env.sig_gyro == 0.002
