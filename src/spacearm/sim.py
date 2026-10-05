"""Free-floating PyBullet simulation of the spacecraft + 7-DOF arm.

* Zero gravity, zero damping everywhere (momentum must not leak), velocity motors on the joints.
* No contact physics (DESIGN D4): collisions are measured geometrically with `getClosestPoints`
  over an explicit pair list (arm x spacecraft, non-adjacent arm x arm, minus the ACM);
  added obstacles are "ghosts" that never exert contact forces on the robot.
* Every PyBullet call passes `physicsClientId`, so several simulators can live in one process.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p

from spacearm.robot_model import ARM_JOINTS, ARM_LINKS, TCP_LINK, build_urdf, joint_limits, spacecraft_links

_IDENTITY_ORN = (0.0, 0.0, 0.0, 1.0)
_OBSTACLE_RGBA = (0.9, 0.15, 0.15, 0.8)


class SpaceRobotSim:
    """One PyBullet client holding the free-floating space robot (base = bus, free 6-DOF)."""

    def __init__(self, cfg: dict, gui: bool = False):
        self.cfg = cfg
        arm = cfg["robot"]["arm"]
        self.max_velocity = float(arm["max_velocity"])
        self.max_torque = float(arm["max_torque"])
        self.max_query_dist = float(cfg["sim"]["max_query_dist"])
        self.dt = 1.0 / cfg["sim"]["physics_hz"]

        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=self.cid)
        p.setTimeStep(self.dt, physicsClientId=self.cid)

        # The URDF is built from the config each time; a private temp folder keeps parallel workers apart
        # (and avoids reopening an open NamedTemporaryFile, which fails on Windows).
        tmp = Path(tempfile.mkdtemp(prefix="spacearm_"))
        try:
            urdf = tmp / "space_robot.urdf"
            urdf.write_text(build_urdf(cfg), encoding="utf-8")
            flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_IMPLICIT_CYLINDER     # no self-collision physics
            self.robot = p.loadURDF(str(urdf), [0, 0, 0], _IDENTITY_ORN, useFixedBase=False, flags=flags,
                                    physicsClientId=self.cid)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # Name -> index maps (never hard-code PyBullet indices). The base link has index -1.
        base_name = p.getBodyInfo(self.robot, physicsClientId=self.cid)[0].decode()
        self.link_index = {base_name: -1}
        self.joint_index, self.parent_of = {}, {}
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            self.joint_index[info[1].decode()] = j
            self.link_index[info[12].decode()] = j          # child link index == joint index
            self.parent_of[j] = info[16]
        self.arm_joint_ids = [self.joint_index[n] for n in ARM_JOINTS]
        self.arm_link_ids = [self.link_index[n] for n in ARM_LINKS]
        self.tcp_link = self.link_index[TCP_LINK]
        self.lower, self.upper = joint_limits(cfg)

        # Zero damping on the base and every link, otherwise momentum leaks (free-floating test fails).
        all_links = [-1] + list(range(p.getNumJoints(self.robot, physicsClientId=self.cid)))
        for k in all_links:
            p.changeDynamics(self.robot, k, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0,
                             physicsClientId=self.cid)
        self.masses = {name: p.getDynamicsInfo(self.robot, k, physicsClientId=self.cid)[0]
                       for name, k in self.link_index.items()}

        # Collision pairs: skip parent-child pairs and the allowed-collision matrix (DESIGN D5).
        skip = {frozenset(pr) for pr in cfg["robot"]["acm_skip"]}

        def keep(a: str, b: str) -> bool:
            ia, ib = self.link_index[a], self.link_index[b]
            return self.parent_of.get(ia) != ib and self.parent_of.get(ib) != ia and frozenset((a, b)) not in skip

        body = spacecraft_links(cfg)
        self.pairs_body = [(self.link_index[a], self.link_index[b]) for a in ARM_LINKS for b in body if keep(a, b)]
        self.pairs_self = [(self.link_index[a], self.link_index[b])
                           for i, a in enumerate(ARM_LINKS) for b in ARM_LINKS[i + 1:] if keep(a, b)]
        self._obstacles: list[int] = []
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, q=None, base_pos=(0.0, 0.0, 0.0), base_orn=_IDENTITY_ORN) -> None:
        """Base at the given pose and at rest, joints at `q` (default zeros) and at rest, motors idle."""
        p.resetBasePositionAndOrientation(self.robot, base_pos, base_orn, physicsClientId=self.cid)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        self.set_q(np.zeros(7) if q is None else q)
        self.apply_joint_velocities(np.zeros(7))

    def set_q(self, q) -> None:
        """Teleport the joints to `q` with zero joint velocity (kinematic query; the base is not moved)."""
        for j, qj in zip(self.arm_joint_ids, np.asarray(q, float)):
            p.resetJointState(self.robot, j, float(qj), 0.0, physicsClientId=self.cid)

    def get_q(self) -> np.ndarray:
        states = p.getJointStates(self.robot, self.arm_joint_ids, physicsClientId=self.cid)
        return np.array([s[0] for s in states])

    def get_qd(self) -> np.ndarray:
        states = p.getJointStates(self.robot, self.arm_joint_ids, physicsClientId=self.cid)
        return np.array([s[1] for s in states])

    def apply_joint_velocities(self, qd) -> None:
        """Velocity-motor targets (rad/s), clipped to the joint speed limit; force capped at max_torque."""
        qd = np.clip(np.asarray(qd, float), -self.max_velocity, self.max_velocity)
        p.setJointMotorControlArray(self.robot, self.arm_joint_ids, p.VELOCITY_CONTROL, targetVelocities=qd.tolist(),
                                    forces=[self.max_torque] * 7, physicsClientId=self.cid)

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            p.stepSimulation(physicsClientId=self.cid)

    # ------------------------------------------------------------------ base and frames
    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), quaternion xyzw (4,)) of the bus centre in the inertial frame."""
        pos, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        return np.array(pos), np.array(orn)

    def base_rotation(self) -> np.ndarray:
        """3x3 rotation body -> world."""
        return np.array(p.getMatrixFromQuaternion(self.base_pose()[1])).reshape(3, 3)

    def base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear, angular) base velocity in the inertial frame."""
        lin, ang = p.getBaseVelocity(self.robot, physicsClientId=self.cid)
        return np.array(lin), np.array(ang)

    def world_to_body(self, x_world) -> np.ndarray:
        """Inertial-frame point(s) (..., 3) -> body-frame point(s)."""
        pos, _ = self.base_pose()
        return (np.asarray(x_world, float) - pos) @ self.base_rotation()

    def link_frame_world(self, link: int) -> tuple[np.ndarray, np.ndarray]:
        """URDF link frame (not the centre of mass) of `link` in the inertial frame: (pos, quat xyzw)."""
        if link == -1:                               # bus frame == its centre of mass (inertial origin at 0)
            return self.base_pose()
        s = p.getLinkState(self.robot, link, computeForwardKinematics=True, physicsClientId=self.cid)
        return np.array(s[4]), np.array(s[5])

    def tcp_world(self) -> np.ndarray:
        return self.link_frame_world(self.tcp_link)[0]

    def tcp_body(self) -> np.ndarray:
        return self.world_to_body(self.tcp_world())

    def tcp_velocity(self) -> np.ndarray:
        """Linear velocity of the TCP in the inertial frame (m/s)."""
        s = p.getLinkState(self.robot, self.tcp_link, computeLinkVelocity=1, computeForwardKinematics=True,
                           physicsClientId=self.cid)
        return np.array(s[6])

    def system_com(self) -> np.ndarray:
        """Centre of mass of the whole system in the inertial frame."""
        total, acc = 0.0, np.zeros(3)
        for name, k in self.link_index.items():
            if k == -1:
                c = self.base_pose()[0]
            else:
                c = np.array(p.getLinkState(self.robot, k, computeForwardKinematics=True,
                                            physicsClientId=self.cid)[0])
            acc += self.masses[name] * c
            total += self.masses[name]
        return acc / total

    # ------------------------------------------------------------------ geometry queries
    def _pair_min(self, pairs, max_dist: float) -> float:
        d = max_dist
        for a, b in pairs:
            for pt in p.getClosestPoints(self.robot, self.robot, max_dist, linkIndexA=a, linkIndexB=b,
                                         physicsClientId=self.cid):
                d = min(d, pt[8])
        return d

    def min_distances(self, max_dist: float | None = None) -> tuple[float, float]:
        """(d_body, d_self): signed minimum distances (m, negative = penetration), clipped at max_dist."""
        max_dist = self.max_query_dist if max_dist is None else float(max_dist)
        p.performCollisionDetection(physicsClientId=self.cid)
        return self._pair_min(self.pairs_body, max_dist), self._pair_min(self.pairs_self, max_dist)

    def in_collision(self, margin: float = 0.0) -> bool:
        """True if the arm is within `margin` of the spacecraft or of itself (margin 0 = touching)."""
        return min(self.min_distances()) <= margin

    # ------------------------------------------------------------------ obstacles (ghost bodies)
    def add_sphere(self, center, r: float) -> int:
        """Static sphere obstacle that never exerts contact forces on the robot (distance queries still work)."""
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(r), physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(r), rgbaColor=_OBSTACLE_RGBA, physicsClientId=self.cid)
        body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                 basePosition=np.asarray(center, float).tolist(), physicsClientId=self.cid)
        for k in self.link_index.values():
            p.setCollisionFilterPair(self.robot, body, k, -1, 0, physicsClientId=self.cid)
        self._obstacles.append(body)
        return body

    def move_body(self, body: int, center) -> None:
        p.resetBasePositionAndOrientation(body, np.asarray(center, float).tolist(), _IDENTITY_ORN,
                                          physicsClientId=self.cid)

    def remove_body(self, body: int) -> None:
        p.removeBody(body, physicsClientId=self.cid)
        if body in self._obstacles:
            self._obstacles.remove(body)

    def body_distance(self, body: int, links=None, max_dist: float | None = None):
        """(d, p_robot, p_body): signed minimum distance between `links` (default: arm links) and `body`,
        with the closest points in the inertial frame. If nothing is within max_dist: (max_dist, None, None)."""
        max_dist = self.max_query_dist if max_dist is None else float(max_dist)
        links = self.arm_link_ids if links is None else links
        p.performCollisionDetection(physicsClientId=self.cid)
        best = (max_dist, None, None)
        for k in links:
            for pt in p.getClosestPoints(self.robot, body, max_dist, linkIndexA=k, linkIndexB=-1,
                                         physicsClientId=self.cid):
                if pt[8] < best[0]:
                    best = (pt[8], np.array(pt[5]), np.array(pt[6]))
        return best

    def close(self) -> None:
        if self.cid is not None and p.isConnected(physicsClientId=self.cid):
            p.disconnect(physicsClientId=self.cid)
        self.cid = None
