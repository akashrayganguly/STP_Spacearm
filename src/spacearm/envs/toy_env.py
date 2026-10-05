"""PointHazardEnv: a tiny constrained task to check PPO-Lagrangian before the expensive arm task (DESIGN §6.3).

A point moves in the plane from the left side to a goal on the right side. A hazard disc sits on the straight
line between them: crossing it costs 1 per step (about 7 per episode on the straight line), going around
costs nothing but a few extra steps. An unconstrained learner cuts through; a constrained one goes around.

Same interface as SpaceReachEnv: action (1, 2) in [-1, 1]; observation {"actor": (1, 6), "critic": (7,)}
with critic = actor features + t/T; info has cost, success, collision, d_tcp, clearance.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Task geometry (toy constants: this env exists only to test the learner).
STEP = 0.05                  # max displacement per step and axis
HAZARD_R = 0.18              # hazard radius at difficulty 1 (straight line: ~7 steps inside)
GOAL_TOL = 0.05              # success radius
START_X, GOAL_X, Y_SPREAD = -0.7, 0.7, 0.3
HAZARD_JITTER = 0.03         # hazard centre offset from the start-goal midpoint
ARENA = 1.0
REWARD = {"progress": 10.0, "time": 0.01, "success": 5.0}


class PointHazardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None, difficulty: float = 1.0, max_steps: int = 60):
        super().__init__()
        self.cfg, self.max_steps = cfg, int(max_steps)
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.action_space = spaces.Box(-1.0, 1.0, (1, 2), np.float32)
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(-np.inf, np.inf, (1, 6), np.float32),
            "critic": spaces.Box(-np.inf, np.inf, (7,), np.float32)})

    def set_difficulty(self, d: float) -> None:
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    @property
    def hazard_r(self) -> float:
        return HAZARD_R * self.difficulty

    def _obs(self) -> dict:
        actor = np.concatenate([self.pos, self.goal - self.pos, self.hazard - self.pos]).astype(np.float32)
        critic = np.concatenate([actor, [self.t / self.max_steps]]).astype(np.float32)
        return {"actor": actor[None], "critic": critic}

    def _info(self, cost: float, success: bool) -> dict:
        d_haz = float(np.linalg.norm(self.pos - self.hazard)) - self.hazard_r
        return {"cost": cost, "success": success, "collision": False,
                "d_tcp": float(np.linalg.norm(self.goal - self.pos)), "clearance": d_haz}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos = np.array([START_X, rng.uniform(-Y_SPREAD, Y_SPREAD)])
        self.goal = np.array([GOAL_X, rng.uniform(-Y_SPREAD, Y_SPREAD)])
        mid = (self.pos + self.goal) / 2
        u = (self.goal - self.pos) / np.linalg.norm(self.goal - self.pos)
        self.hazard = mid + np.array([-u[1], u[0]]) * rng.uniform(-HAZARD_JITTER, HAZARD_JITTER)
        self.t = 0
        return self._obs(), self._info(0.0, False)

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(-1)[:2], -1.0, 1.0)
        d_prev = np.linalg.norm(self.goal - self.pos)
        self.pos = np.clip(self.pos + STEP * a, -ARENA, ARENA)
        self.t += 1
        d = np.linalg.norm(self.goal - self.pos)
        success = bool(d < GOAL_TOL)
        cost = 1.0 if np.linalg.norm(self.pos - self.hazard) < self.hazard_r else 0.0
        reward = REWARD["progress"] * (d_prev - d) - REWARD["time"] + REWARD["success"] * success
        truncated = bool(self.t >= self.max_steps and not success)
        return self._obs(), float(reward), success, truncated, self._info(cost, success)
