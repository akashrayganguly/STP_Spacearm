"""Phase 1b - exact differentiable forward kinematics (PyTorch) agrees with PyBullet."""
import numpy as np
import pybullet as p
import torch

LINKS = [f"link{i}" for i in range(1, 8)] + ["tcp"]


def test_fk_positions_and_rotations_match_pybullet(sim, kin64, rng):
    for _ in range(30):
        q = rng.uniform(sim.lower, sim.upper)
        sim.set_q(q)                                      # base at identity -> world == body frame
        R, t = kin64.forward(torch.tensor(q))
        for k, name in enumerate(LINKS):
            pos, orn = sim.link_frame_world(sim.link_index[name])
            np.testing.assert_allclose(t[k].numpy(), pos, atol=1e-5)
            np.testing.assert_allclose(R[k].numpy(), np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3),
                                       atol=1e-5)


def test_link_frame_is_not_centre_of_mass(sim):
    """Pitfall guard: getLinkState()[0] is the CoM, [4] is the URDF link frame used by FK."""
    sim.set_q(np.zeros(7))
    s = p.getLinkState(sim.robot, sim.link_index["link3"], computeForwardKinematics=True, physicsClientId=sim.cid)
    assert np.linalg.norm(np.array(s[0]) - np.array(s[4])) > 0.05


def test_tcp_at_zero_configuration(cfg, kin64):
    r, a = cfg["robot"], cfg["robot"]["arm"]
    height = r["pedestal"]["length"] + sum(a["link_lengths"][:6]) + a["tool_offset"]
    expected = np.array(r["mount_xyz"]) + np.array([0.0, 0.0, height])
    np.testing.assert_allclose(kin64.tcp(torch.zeros(7)).numpy(), expected, atol=1e-12)


def test_batched_shapes(kin64):
    q = torch.zeros(4, 3, 7, dtype=torch.float64)
    R, t = kin64.forward(q)
    assert R.shape == (4, 3, 8, 3, 3) and t.shape == (4, 3, 8, 3)
    assert kin64.tcp(q).shape == (4, 3, 3)
    assert kin64.jacobian(q).shape == (4, 3, 3, 7)


def test_analytic_jacobian_matches_autograd(kin64, rng):
    for _ in range(5):
        q = torch.tensor(rng.uniform(-2.0, 2.0, 7))
        J_auto = torch.autograd.functional.jacobian(kin64.tcp, q)
        torch.testing.assert_close(kin64.jacobian(q), J_auto, atol=1e-10, rtol=0)


def test_capsule_segments(cfg, kin64, rng):
    q = torch.tensor(rng.uniform(-2.0, 2.0, (5, 7)))
    a, b, r = kin64.capsules(q)
    assert a.shape == (5, 7, 3) and b.shape == (5, 7, 3) and r.shape == (7,)
    np.testing.assert_allclose(torch.linalg.norm(b - a, dim=-1).numpy(),
                               np.broadcast_to(cfg["robot"]["arm"]["link_lengths"], (5, 7)), atol=1e-12)


def test_fk_is_differentiable(kin64):
    q = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    kin64.tcp(q).sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
