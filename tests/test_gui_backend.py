"""v1.1 - GUI backend (no browser needed): the numbers the app shows come from the same code as the evaluations."""
import numpy as np
import pytest

pytest.importorskip("plotly")

from conftest import model_file  # noqa: E402


@pytest.fixture(scope="module")
def B(cfg):
    for name in ("distance_net.pt", "traj_net.pt", "ppo_lag.pt"):
        model_file(cfg, name)
    from spacearm.gui import backend

    return backend


def test_pose_report_matches_the_simulator(B):
    q_deg = [10.0, 40.0, -20.0, 70.0, 15.0, 30.0, 0.0]
    rep = B.pose_report(q_deg)
    m = B.models()
    m.sim.set_q(np.radians(q_deg))
    d_body, d_self = m.sim.min_distances()
    assert rep["d_body"] == pytest.approx(d_body) and rep["d_self"] == pytest.approx(d_self)
    assert abs(rep["pred_body"] - d_body) < 0.05            # DistanceNet within 5 cm on an ordinary pose
    assert rep["status"] == ("SAFE" if min(d_body, d_self) >= 0.05 else rep["status"])
    fig = B.pose_figure(rep)
    assert len(fig.data) >= 2


def test_scene_meshes_are_valid(B):
    from spacearm.gui import scene

    V, F, C = scene.arm_mesh(B.models().kin, np.zeros(7))
    assert F.min() >= 0 and F.max() < len(V) and len(C) == len(F)
    V, F, C = scene.spacecraft_mesh(B.models().cfg)
    assert F.max() < len(V) and np.isfinite(V).all()


def test_check_target(B):
    shoulder, reach = B.shoulder_and_reach()
    assert B.check_target(shoulder + [0.0, 0.0, 0.5])[0]
    assert not B.check_target(shoulder + [0.0, 0.0, reach + 0.1])[0]      # out of reach
    assert not B.check_target([0.0, 0.0, 0.0])[0]                        # inside the bus


def test_mission_records_a_consistent_episode(B):
    shoulder, _ = B.shoulder_and_reach()
    spec = B.MissionSpec(seed=3, obstacle="static", radius_cm=7, appear_s=1.0, bias_deg=1.0, motor_min=0.8,
                         slip_s=None, noise=0.5, obstacle_center=(shoulder + [0.0, 0.0, 1.3]).tolist())
    tr = B.run_mission("Reflex only", spec, render=False)
    n = len(tr["t"])
    assert tr["outcome"] in ("SUCCESS", "COLLISION", "TIME-OUT")
    assert all(len(tr[k]) == n for k in ("q", "base_p", "base_q", "tcp", "d_tcp", "shield"))
    assert max(abs(b) for b in tr["sampled"]["bias_deg"]) <= 1.0 + 1e-9
    assert min(tr["sampled"]["gains"]) >= 0.8 - 1e-9
    assert tr["sampled"]["obstacle_s"] == pytest.approx(1.0)
    if n > 11:                                              # a pinned ball exists from t = 1 s (unless already done)
        assert tr["sampled"]["obstacle_seen_s"] == pytest.approx(1.0)
        np.testing.assert_allclose(tr["obs_center"][11], shoulder + [0.0, 0.0, 1.3])
