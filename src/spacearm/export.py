"""Deployment exports of the actor (DESIGN §6.5, D16): observation normalisation baked in.

* NumPy .npz: three matrix multiplies + tanh, no PyTorch needed at run time (~10 us per call).
* ONNX (opset 17): input "obs" (batch, obs_dim) float32 -> output "action" (batch, act_dim).
Both reproduce `PPOLagAgent.act` (the deterministic policy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from spacearm.rl.ppo_lag import _EPS, OBS_CLIP

ONNX_OPSET = 17


def _linear_layers(agent) -> list[nn.Linear]:
    return [m for m in agent.actor.net if isinstance(m, nn.Linear)]


def export_actor_numpy(agent, path) -> Path:
    """Write the deterministic actor (normaliser + MLP weights) to an .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"obs_mean": agent.actor_rms.mean.astype(np.float32),
              "obs_std": np.sqrt(agent.actor_rms.var + _EPS).astype(np.float32),
              "obs_clip": np.float32(OBS_CLIP)}
    layers = _linear_layers(agent)
    for i, lin in enumerate(layers):
        arrays[f"W{i}"] = lin.weight.detach().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32)
    arrays["n_layers"] = np.int64(len(layers))
    np.savez(path, **arrays)
    return path


class NumpyActor:
    """Deterministic policy from an .npz export: obs (..., obs_dim) -> action (..., act_dim) in [-1, 1]."""

    def __init__(self, path):
        with np.load(Path(path)) as f:
            self.mean, self.std, self.clip = f["obs_mean"], f["obs_std"], float(f["obs_clip"])
            n = int(f["n_layers"])
            self.W = [np.ascontiguousarray(f[f"W{i}"].T) for i in range(n)]      # (in, out) for x @ W
            self.b = [f[f"b{i}"] for i in range(n)]

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def __call__(self, obs) -> np.ndarray:
        x = np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -self.clip, self.clip)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = np.tanh(x @ W + b)
        return np.tanh(x @ self.W[-1] + self.b[-1])

    def act(self, obs) -> np.ndarray:            # same interface as PPOLagAgent (usable by evaluate)
        return self(obs)


class _DeployedActor(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.register_buffer("mean", torch.tensor(agent.actor_rms.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.sqrt(agent.actor_rms.var + _EPS), dtype=torch.float32))
        self.net = agent.actor.net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.clamp((obs - self.mean) / self.std, -OBS_CLIP, OBS_CLIP)))


def export_actor_onnx(agent, path) -> Path:
    """Write the deterministic actor with its normaliser to ONNX (dynamic batch dimension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = _DeployedActor(agent).eval()
    dummy = torch.zeros(1, agent.obs_dim, dtype=torch.float32)
    torch.onnx.export(model, (dummy,), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=ONNX_OPSET,
                      dynamo=False)                         # legacy exporter (lesson 19; dynamo needs onnxscript)
    return path
