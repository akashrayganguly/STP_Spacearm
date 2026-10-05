"""Shared fixtures. Run one phase at a time, e.g.  pytest tests/test_p1_robot_sim.py -v"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from spacearm.config import deep_update, load_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)          # small networks: one thread is fast and avoids CPU oversubscription


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture
def sim(cfg):
    from spacearm.sim import SpaceRobotSim

    s = SpaceRobotSim(cfg)
    yield s
    s.close()


@pytest.fixture(scope="session")
def kin64(cfg):
    from spacearm.kinematics import ArmKinematics

    return ArmKinematics(cfg, dtype=torch.float64)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def test_env_cfg(cfg):
    """Deterministic Level-2 config: PyBullet self-clearance, no noise, no faults, no obstacles."""
    return deep_update(cfg, {"env": {"self_clearance": "sim", "obstacle": {"prob": 0.0},
                                     "faults": {"enabled": False}, "noise": {"enabled": False}}})


def model_file(cfg, name: str) -> Path:
    """Path to a trained model; skips the calling test if it has not been trained yet."""
    path = ROOT / cfg["paths"]["models_dir"] / name
    if not path.exists():
        pytest.skip(f"{path.name} not trained yet")
    return path
