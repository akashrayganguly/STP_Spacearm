"""Level-2 environment (DESIGN §6.1): reach an inertially fixed target with the 7-DOF arm on a free-floating
spacecraft, while surprise obstacles appear and joint hardware misbehaves.

* Action (1, 7) in [-1, 1]. Residual mode: command = clip(prior + residual_scale * fade * action),
  fade = min(1, measured distance to target / residual_fade); joint velocity = command * v_max * actuator gain.
* Observation {"actor": (1, 56) onboard-measurable features, "critic": (90,) = actor + 34 privileged values}.
* Cost 1 when the true clearance (body, self or obstacle) < collision.cost_dist; termination on collision
  (clearance <= 0) or success (TCP within success_dist and slower than success_speed); truncation at max_steps.
* Difficulty d in [0, 1] scales obstacle probability, faults and sensor noise.

The dynamic simulator is never teleported during an episode (set_q would zero the velocities): model-based
features (FK, DistanceNet, obstacle clearance) are computed from the *measured* joint angles instead.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pybullet as p
import torch
from gymnasium import spaces

from spacearm.config import ROOT
from spacearm.control import reach_prior
from spacearm.kinematics import ArmKinematics
from spacearm.safety.shield import sphere_capsule_distance
from spacearm.sim import SpaceRobotSim

# Observation layout (DESIGN §6.1). Scales are part of the layout definition.
GYRO_SCALE = 0.1             # rad/s
CLEAR_SCALE = 0.1            # m
SL = {"q": slice(0, 7), "qd": slice(7, 14), "rel": slice(14, 17), "dist": slice(17, 18), "gyro": slice(18, 21),
      "obs_rel": slice(21, 24), "obs_r": slice(24, 25), "obs_active": slice(25, 26),
      "obs_clear": slice(26, 27), "obs_grad": slice(27, 34), "self_clear": slice(34, 35),
      "self_grad": slice(35, 42), "prev_cmd": slice(42, 49), "prior": slice(49, 56)}
ACTOR_DIM = 56
# Privileged critic block (after the actor features): true q (7), encoder bias (7), actuator gains (7),
# base quaternion (4), base linear velocity (3), true self clearance (1), true obstacle clearance (1),
# obstacle velocity (3), t / T (1).
CRITIC_DIM = ACTOR_DIM + 34
_FREE_TRIES = 1000           # rejection-sampling cap for collision-free start / target configurations
_SPAWN_TRIES = 20            # tries to place an obstacle that does not start inside the arm or the target
_SPAWN_FRAC = (0.25, 0.75)   # obstacle centre along the TCP -> target segment at spawn time
_CAMERA = {"distance": 3.3, "yaw": 40.0, "pitch": -22.0, "target_z": 0.55}


def shaped_reward(d_prev: float, d: float, a, a_prev, success: bool, w: dict) -> float:
    """Potential-based progress - action rate - time + success - residual size (DESIGN §6.1)."""
    a, a_prev = np.asarray(a, float), np.asarray(a_prev, float)
    return float(w["progress"] * (d_prev - d) - w["action_rate"] * np.sum((a - a_prev) ** 2) - w["time"]
                 + w["success"] * float(success) - w["residual"] * np.sum(a ** 2))


def _unit(g: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(g)
    return g / n if n > 1e-9 else np.zeros_like(g)


class SpaceReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: dict, difficulty: float = 1.0, gui: bool = False):
        super().__init__()
        self.cfg = cfg
        self.e = cfg["env"]
        self.sim = SpaceRobotSim(cfg, gui=gui)
        self.kin = ArmKinematics(cfg, dtype=torch.float64)
        self.n_sub = int(round(cfg["sim"]["physics_hz"] / cfg["sim"]["control_hz"]))
        self.dt = 1.0 / cfg["sim"]["control_hz"]
        self.v_max = float(cfg["robot"]["arm"]["max_velocity"])
        self.max_steps = int(self.e["max_steps"])
        self.clip_dist = float(cfg["sim"]["max_query_dist"])
        self.mid = (self.sim.lower + self.sim.upper) / 2
        self.half = (self.sim.upper - self.sim.lower) / 2
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.shield = None
        self.dnet = None
        if self.e["self_clearance"] == "net":
            from spacearm.models.distance_net import DistanceNet
            torch.set_num_threads(1)                         # env workers: one thread (lesson 18)
            self.dnet = DistanceNet.from_config(cfg)
            self.dnet.load_state_dict(torch.load(ROOT / self.e["distance_net_path"], map_location="cpu"))
            self.dnet.eval().requires_grad_(False)
        elif self.e["self_clearance"] != "sim":
            raise ValueError(f"env.self_clearance must be 'net' or 'sim', got {self.e['self_clearance']!r}")
        self.action_space = spaces.Box(-1.0, 1.0, (1, 7), np.float32)
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(-np.inf, np.inf, (1, ACTOR_DIM), np.float32),
            "critic": spaces.Box(-np.inf, np.inf, (CRITIC_DIM,), np.float32)})
        self.render_mode = "rgb_array"
        self._marker = None
        self.obstacle = None

    # ------------------------------------------------------------------ configuration
    def set_difficulty(self, d: float) -> None:
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def set_shield(self, shield) -> None:
        """Filter every final command through `shield.filter(q_meas, cmd, obstacle)` (None to remove)."""
        self.shield = shield

    def place_obstacle(self, center_w, radius: float, velocity=None) -> None:
        """Put a (ghost) sphere obstacle at an inertial position now, replacing any existing one."""
        self._remove_obstacle()
        center_w = np.asarray(center_w, float)
        self.obstacle = {"body": self.sim.add_sphere(center_w, radius), "center": center_w.copy(),
                         "radius": float(radius), "vel": np.zeros(3) if velocity is None else np.asarray(velocity, float)}
        self._true = self._true_clearances()

    def _remove_obstacle(self) -> None:
        if self.obstacle is not None:
            self.sim.remove_body(self.obstacle["body"])
        self.obstacle = None

    # ------------------------------------------------------------------ sampling helpers
    def _free_config(self, rng) -> np.ndarray:
        frac, margin = self.cfg["data"]["limit_frac"], self.cfg["collision"]["plan_margin"]
        for _ in range(_FREE_TRIES):
            q = rng.uniform(self.mid - frac * self.half, self.mid + frac * self.half)
            self.sim.set_q(q)
            if min(self.sim.min_distances()) > margin:
                return q
        raise RuntimeError("could not sample a collision-free configuration")

    def _tcp_fk(self, q) -> np.ndarray:
        with torch.no_grad():
            return self.kin.tcp(torch.as_tensor(q, dtype=torch.float64)).numpy()

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng, d, e = self.np_random, self.difficulty, self.e
        self._remove_obstacle()
        self.sim.reset()
        q_start = self._free_config(rng)
        tcp_start = self._tcp_fk(q_start)
        while True:
            q_t = self._free_config(rng)
            target = self._tcp_fk(q_t)
            if np.linalg.norm(target - tcp_start) >= self.cfg["traj"]["min_target_dist"]:
                break
        self.sim.reset(q=q_start)                          # base at the origin: body frame == inertial frame
        self.target_w = target

        f, n = e["faults"], e["noise"]
        self.bias = np.zeros(7)
        self.gains = np.ones(7)
        self.slip_step, self.slip = -1, np.zeros(7)
        if f["enabled"]:
            self.bias = np.radians(rng.uniform(-f["bias_deg"] * d, f["bias_deg"] * d, 7))
            self.gains = rng.uniform(1.0 - (1.0 - f["gain_min"]) * d, 1.0, 7)
            if rng.random() < f["slip_prob"] * d:
                self.slip_step = int(rng.integers(1, self.max_steps))
                self.slip[rng.integers(0, 7)] = np.radians(f["slip_deg"]) * rng.choice([-1.0, 1.0])
        self.sig_enc = np.radians(n["encoder_deg"]) * d if n["enabled"] else 0.0
        self.sig_vis = n["vision_m"] * d if n["enabled"] else 0.0
        self.sig_gyro = n["gyro"] * d if n["enabled"] else 0.0

        o = e["obstacle"]
        self.spawn_step = -1
        if rng.random() < o["prob"] * d:
            self.spawn_step = int(round(rng.uniform(*o["spawn_time"]) / self.dt))
            self.spawn_radius = rng.uniform(*o["radius"])
            moving = rng.random() < o["move_prob"]
            direction = _unit(rng.normal(size=3))
            self.spawn_vel = direction * rng.uniform(0.0, o["max_speed"]) if moving else np.zeros(3)

        self.t = 0
        self.prev_cmd = np.zeros(7)
        self.prev_action = np.zeros(7)
        self._true = self._true_clearances()
        self.d_tcp = float(np.linalg.norm(self.sim.tcp_world() - self.target_w))
        obs = self._observe()
        return obs, self._info(0.0, False, False)

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(-1)[:7], -1.0, 1.0)
        e = self.e
        if e["residual"]:
            fade = min(1.0, self._d_meas / e["residual_fade"])
            cmd = np.clip(self._prior + e["residual_scale"] * fade * a, -1.0, 1.0)
        else:
            cmd = a.copy()
        shield_info = None
        if self.shield is not None:
            cmd, shield_info = self.shield.filter(self._q_meas, cmd, self._obstacle_estimate())
            cmd = np.clip(np.asarray(cmd, float), -1.0, 1.0)

        self.sim.apply_joint_velocities(cmd * self.v_max * self.gains)
        self.sim.step(self.n_sub)
        self.t += 1
        if self.obstacle is not None and np.any(self.obstacle["vel"]):
            self.obstacle["center"] += self.obstacle["vel"] * self.dt
            self.sim.move_body(self.obstacle["body"], self.obstacle["center"])
        if self.t == self.slip_step:
            self.bias = self.bias + self.slip
        if self.t == self.spawn_step and self.obstacle is None:
            self._spawn_obstacle()

        self._true = self._true_clearances()
        d_prev = self.d_tcp
        self.d_tcp = float(np.linalg.norm(self.sim.tcp_world() - self.target_w))
        clearance = self._true["min"]
        collision = bool(clearance <= 0.0)
        speed = float(np.linalg.norm(self.sim.tcp_velocity()))
        success = bool(not collision and self.d_tcp < e["success_dist"] and speed < e["success_speed"])
        cost = 1.0 if clearance < self.cfg["collision"]["cost_dist"] else 0.0
        reward = shaped_reward(d_prev, self.d_tcp, a, self.prev_action, success, e["reward"])
        terminated = collision or success
        truncated = bool(self.t >= self.max_steps and not terminated)
        self.prev_cmd, self.prev_action = cmd, a
        obs = self._observe()
        info = self._info(cost, success, collision)
        info.update(cmd=cmd, tcp_speed=speed)
        if shield_info is not None:
            info["shield"] = shield_info
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    # ------------------------------------------------------------------ internals
    def _spawn_obstacle(self) -> None:
        """Sphere between the TCP and the target (+-5 cm), not inside the arm, not covering the target."""
        rng, margin = self.np_random, self.cfg["collision"]["plan_margin"]
        tcp = self.sim.tcp_world()
        for _ in range(_SPAWN_TRIES):
            c = tcp + rng.uniform(*_SPAWN_FRAC) * (self.target_w - tcp) + rng.uniform(-0.05, 0.05, 3)
            if np.linalg.norm(c - self.target_w) < self.spawn_radius + margin:
                continue
            self.place_obstacle(c, self.spawn_radius, self.spawn_vel)
            if self._true["obs"] > margin:
                return
            self._remove_obstacle()

    def _true_clearances(self) -> dict:
        d_body, d_self = self.sim.min_distances()
        d_obs = self.clip_dist
        if self.obstacle is not None:
            d_obs = min(self.clip_dist, self.sim.body_distance(self.obstacle["body"])[0])
        return {"body": d_body, "self": d_self, "obs": d_obs, "min": min(d_body, d_self, d_obs)}

    def _obstacle_estimate(self):
        """Obstacle as the robot sees it (body frame, from vision relative to the FK of the measured TCP)."""
        if self.obstacle is None:
            return None
        return {"center": self._obs_center_est, "radius": self.obstacle["radius"]}

    def _observe(self) -> dict:
        rng = self.np_random
        q_true, qd = self.sim.get_q(), self.sim.get_qd()
        q_meas = q_true + self.bias + rng.normal(0.0, self.sig_enc, 7)
        pos, orn = self.sim.base_pose()
        R = self.sim.base_rotation()
        lin_vel, ang_vel = self.sim.base_velocity()
        tcp_w = self.sim.tcp_world()
        rel = R.T @ (self.target_w - tcp_w) + rng.normal(0.0, self.sig_vis, 3)
        gyro = R.T @ ang_vel + rng.normal(0.0, self.sig_gyro, 3)

        # One exact FK pass on the measured angles serves the prior's Jacobian and the obstacle features.
        active = self.obstacle is not None
        qt = torch.tensor(q_meas, dtype=torch.float64, requires_grad=active)
        Rk, tk = self.kin.forward(qt)
        J = self.kin.jacobian_from(Rk, tk).detach().numpy()

        # Obstacle: vision gives its centre relative to the TCP; the robot places it with FK(q_meas).
        obs_rel, obs_r = np.zeros(3), 0.0
        obs_clear, obs_grad = self.clip_dist, np.zeros(7)
        obs_vel = np.zeros(3)
        if active:
            obs_rel = R.T @ (self.obstacle["center"] - tcp_w) + rng.normal(0.0, self.sig_vis, 3)
            obs_r, obs_vel = self.obstacle["radius"], self.obstacle["vel"]
            self._obs_center_est = tk[-1].detach().numpy() + obs_rel
            a, b, rad = self.kin.capsules_from(Rk, tk)
            c = sphere_capsule_distance(torch.tensor(self._obs_center_est), obs_r, a, b, rad).min()
            c.backward()
            obs_clear, obs_grad = min(self.clip_dist, c.item()), _unit(qt.grad.numpy())

        # Self clearance (spacecraft body + arm itself): DistanceNet on measured angles, or PyBullet truth.
        if self.dnet is not None:
            qt = torch.tensor(q_meas, dtype=torch.float32, requires_grad=True)
            c = self.dnet.clearance(qt)
            c.backward()
            self_clear, self_grad = min(self.clip_dist, c.item()), _unit(qt.grad.double().numpy())
        else:
            self_clear, self_grad = min(self._true["body"], self._true["self"]), np.zeros(7)

        self._q_meas, self._d_meas = q_meas, float(np.linalg.norm(rel))
        self._prior = reach_prior(self.kin, q_meas, rel, self.cfg, clearance=self_clear, clearance_grad=self_grad,
                                  obstacle_clearance=obs_clear if active else None,
                                  obstacle_grad=obs_grad if active else None, J=J)
        actor = np.concatenate([
            (q_meas - self.mid) / self.half, qd / self.v_max, rel, [self._d_meas], gyro / GYRO_SCALE,
            obs_rel, [obs_r], [float(active)], [obs_clear / CLEAR_SCALE], obs_grad,
            [self_clear / CLEAR_SCALE], self_grad, self.prev_cmd, self._prior])
        priv = np.concatenate([
            (q_true - self.mid) / self.half, self.bias, self.gains, orn, lin_vel,
            [min(self._true["body"], self._true["self"]) / CLEAR_SCALE], [self._true["obs"] / CLEAR_SCALE],
            obs_vel, [self.t / self.max_steps]])
        critic = np.concatenate([actor, priv])
        return {"actor": actor.astype(np.float32)[None], "critic": critic.astype(np.float32)}

    def _info(self, cost: float, success: bool, collision: bool) -> dict:
        _, orn = self.sim.base_pose()
        return {"cost": cost, "success": success, "collision": collision, "d_tcp": self.d_tcp,
                "clearance": self._true["min"], "d_body": self._true["body"], "d_self": self._true["self"],
                "d_obs": self._true["obs"], "obstacle_active": self.obstacle is not None, "t": self.t,
                "base_rotation_deg": float(np.degrees(2 * np.arccos(min(1.0, abs(float(orn[3])))))),
                "prior": self._prior.copy()}

    # ------------------------------------------------------------------ rendering (offscreen)
    def render(self, width: int = 320, height: int = 240) -> np.ndarray:
        """RGB frame (H, W, 3) from a camera that follows the spacecraft; the target is a green dot."""
        cid = self.sim.cid
        if self._marker is None:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0.1, 0.8, 0.2, 1.0], physicsClientId=cid)
            self._marker = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=-1,
                                             basePosition=self.target_w.tolist(), physicsClientId=cid)
        p.resetBasePositionAndOrientation(self._marker, self.target_w.tolist(), [0, 0, 0, 1], physicsClientId=cid)
        pos, _ = self.sim.base_pose()
        look = [pos[0], pos[1], pos[2] + _CAMERA["target_z"]]
        view = p.computeViewMatrixFromYawPitchRoll(look, _CAMERA["distance"], _CAMERA["yaw"], _CAMERA["pitch"], 0.0, 2,
                                                   physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(45.0, width / height, 0.05, 20.0, physicsClientId=cid)
        _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                                            lightDirection=[1.0, -1.0, 2.0], shadow=0, physicsClientId=cid)
        return np.reshape(np.asarray(rgba, np.uint8), (height, width, 4))[:, :, :3]
