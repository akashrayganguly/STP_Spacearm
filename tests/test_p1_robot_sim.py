"""Phase 1a - robot model and free-floating simulator."""
import numpy as np
import pybullet as p
import pytest

from spacearm.robot_model import ARM_JOINTS, build_urdf, joint_limits, spacecraft_links, total_mass

# Arm folded back over the payload: upper arm points -x, forearm points down into the deck.
Q_FOLDED = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0])


def substeps(cfg):
    return int(round(cfg["sim"]["physics_hz"] / cfg["sim"]["control_hz"]))


def test_urdf_text_has_seven_revolute_joints_and_capsules(cfg):
    txt = build_urdf(cfg)
    assert txt.count('type="revolute"') == 7
    for name in ARM_JOINTS:
        assert f'name="{name}"' in txt
    assert "<capsule" in txt


def test_loads_with_named_revolute_joints(sim):
    assert len(sim.arm_joint_ids) == 7
    for j in sim.arm_joint_ids:
        assert p.getJointInfo(sim.robot, j, physicsClientId=sim.cid)[2] == p.JOINT_REVOLUTE
    assert sim.link_index["bus"] == -1
    assert "tcp" in sim.link_index


def test_joint_limits_match_config(cfg, sim):
    lo, hi = joint_limits(cfg)
    for k, j in enumerate(sim.arm_joint_ids):
        info = p.getJointInfo(sim.robot, j, physicsClientId=sim.cid)
        assert info[8] == pytest.approx(lo[k], abs=1e-5)
        assert info[9] == pytest.approx(hi[k], abs=1e-5)


def test_total_mass_matches_config(cfg, sim):
    assert sum(sim.masses.values()) == pytest.approx(total_mass(cfg), abs=1e-6)


def test_zero_configuration_is_clear(sim):
    sim.set_q(np.zeros(7))
    d_body, d_self = sim.min_distances()
    assert d_body > 0.10 and d_self > 0.10


def test_folded_configuration_collides_with_spacecraft(sim):
    sim.set_q(Q_FOLDED)
    d_body, _ = sim.min_distances()
    assert d_body < 0.0
    assert sim.in_collision()


def test_collision_pairs_respect_parent_child_and_acm(cfg, sim):
    names = {v: k for k, v in sim.link_index.items()}
    skip = {frozenset(pr) for pr in cfg["robot"]["acm_skip"]}
    for a, b in sim.pairs_body + sim.pairs_self:
        assert sim.parent_of.get(a) != b and sim.parent_of.get(b) != a
        assert frozenset((names[a], names[b])) not in skip
    as_names = {frozenset((names[a], names[b])) for a, b in sim.pairs_body + sim.pairs_self}
    assert frozenset(("link4", "bus")) in as_names
    assert frozenset(("link5", "payload")) in as_names
    assert frozenset(("link1", "link7")) in as_names
    # default geometry: 7 arm links x 5 spacecraft parts - 1 parent pair - 4 ACM pairs = 30;
    # 21 arm-arm pairs - 6 adjacent - 5 ACM pairs = 10   (update if you change the geometry)
    assert len(sim.pairs_body) == 30 and len(sim.pairs_self) == 10
    assert set(spacecraft_links(cfg)) >= {"bus", "panel_left", "panel_right", "pedestal"}


def test_acm_pairs_never_collide(cfg, sim, rng):
    """Every skipped pair must be impossible to collide within the joint limits (sampled check)."""
    pairs = [(sim.link_index[a], sim.link_index[b]) for a, b in cfg["robot"]["acm_skip"]]
    worst = np.inf
    for _ in range(300):
        sim.set_q(rng.uniform(sim.lower, sim.upper))
        p.performCollisionDetection(physicsClientId=sim.cid)
        for a, b in pairs:
            for pt in p.getClosestPoints(sim.robot, sim.robot, 1.0, linkIndexA=a, linkIndexB=b,
                                         physicsClientId=sim.cid):
                worst = min(worst, pt[8])
    assert worst > 0.0


def test_set_get_q_roundtrip(sim, rng):
    q = rng.uniform(sim.lower, sim.upper)
    sim.set_q(q)
    np.testing.assert_allclose(sim.get_q(), q, atol=1e-9)


def test_free_floating_conserves_momentum_and_base_reacts(cfg, sim):
    sim.reset()
    com0 = sim.system_com()
    _, orn0 = sim.base_pose()
    v = np.array([0.5, -0.4, 0.3, 0.5, -0.5, 0.4, 0.5])
    for _ in range(20):                                   # 2 s at 10 Hz
        sim.apply_joint_velocities(v)
        sim.step(substeps(cfg))
    _, orn1 = sim.base_pose()
    angle_deg = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(orn0, orn1))))))
    assert np.linalg.norm(sim.system_com() - com0) < 2e-3   # no external force -> CoM fixed
    assert angle_deg > 0.5                                    # spacecraft rotates in reaction
    np.testing.assert_allclose(sim.get_q(), v * 2.0, atol=0.02)  # velocity motors track


def test_base_stops_when_arm_stops(cfg, sim):
    sim.reset()
    for _ in range(10):
        sim.apply_joint_velocities(np.full(7, 0.4))
        sim.step(substeps(cfg))
    for _ in range(5):
        sim.apply_joint_velocities(np.zeros(7))
        sim.step(substeps(cfg))
    _, ang = sim.base_velocity()
    assert np.linalg.norm(ang) < 1e-3                    # zero total momentum -> base at rest again
