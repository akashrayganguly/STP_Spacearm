"""Export the trained actor for deployment and measure latencies on ONE CPU thread (DESIGN §6.5).

  models/actor.npz, models/actor.onnx       deterministic actor with the observation normaliser baked in
  reports/level2/latency.json, latency.md   p50 / p99 of actor (NumPy, ONNX, PyTorch), prior, shield, full env step

    python scripts/export_policy.py [--config ...] [--policy models/ppo_lag.pt] [--n 2000]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from spacearm.config import ROOT, load_config
from spacearm.control import reach_prior
from spacearm.envs.space_reach_env import SL, SpaceReachEnv
from spacearm.export import NumpyActor, export_actor_numpy, export_actor_onnx
from spacearm.models.distance_net import DistanceNet
from spacearm.rl.ppo_lag import PPOLagAgent
from spacearm.safety.shield import shield_from_config


def timeit(fn, n: int, warmup: int = 50) -> dict:
    for _ in range(warmup):
        fn()
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    return {"p50_ms": float(np.percentile(t, 50) * 1e3), "p99_ms": float(np.percentile(t, 99) * 1e3),
            "mean_ms": float(t.mean() * 1e3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default="models/ppo_lag.pt")
    ap.add_argument("--n", type=int, default=2000, help="timed calls per component")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = load_config(args.config)
    models = ROOT / cfg["paths"]["models_dir"]
    agent = PPOLagAgent.load(ROOT / args.policy)
    npz = export_actor_numpy(agent, models / "actor.npz")
    onnx_path = export_actor_onnx(agent, models / "actor.onnx")
    np_actor = NumpyActor(npz)
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])

    # A realistic state: full-difficulty episode, a few steps in, an obstacle placed near the arm.
    env = SpaceReachEnv(cfg, difficulty=1.0)
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(ROOT / cfg["env"]["distance_net_path"], map_location="cpu"))
    shield = shield_from_config(env.kin, cfg, dnet)
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, *_ = env.step(agent.act(obs["actor"]))
    env.place_obstacle(env.sim.tcp_world() + np.array([0.0, 0.0, 0.15]), 0.08)
    obs, *_ = env.step(agent.act(obs["actor"]))
    o = obs["actor"].astype(np.float32)                      # (1, 56)
    a = agent.act(o)[0]
    q_meas, rel = env._q_meas, obs["actor"][0, SL["rel"]].astype(float)
    obstacle = env._obstacle_estimate()

    check = {"numpy_vs_torch_max_abs": float(np.abs(np_actor(o) - agent.act(o)).max()),
             "onnx_vs_torch_max_abs": float(np.abs(sess.run(None, {"obs": o})[0] - agent.act(o)).max())}
    lat = {
        "actor_numpy": timeit(lambda: np_actor(o), args.n),
        "actor_onnx": timeit(lambda: sess.run(None, {"obs": o}), args.n),
        "actor_torch": timeit(lambda: agent.act(o), args.n),
        "prior": timeit(lambda: reach_prior(env.kin, q_meas, rel, cfg), args.n),
        "shield_pass": timeit(lambda: shield.filter(q_meas, np.zeros(7), None), args.n),
        "shield_obstacle": timeit(lambda: shield.filter(q_meas, a, obstacle), args.n),
    }
    shield_case = shield.filter(q_meas, a, obstacle)[1]
    env.set_shield(shield)

    def full_step():                                          # observation features + policy + prior + shield + physics
        o2, _, te, tr, _ = env.step(np_actor(full_step.obs["actor"]))
        full_step.obs = o2 if not (te or tr) else env.reset(seed=124)[0]

    full_step.obs = obs
    lat["env_step_with_shield"] = timeit(full_step, max(200, args.n // 10), warmup=10)
    env.close()

    params = {"actor": int(sum(p.numel() for p in agent.actor.parameters())), "actor_npz": np_actor.n_params,
              "critic_reward": int(sum(p.numel() for p in agent.critic_r.parameters())),
              "distance_net": int(sum(p.numel() for p in dnet.parameters()))}
    res = {"latency": lat, "params": params, "export_check": check, "threads": 1,
           "files": {"npz": str(npz.relative_to(ROOT)), "onnx": str(onnx_path.relative_to(ROOT)),
                     "npz_kb": npz.stat().st_size / 1024, "onnx_kb": onnx_path.stat().st_size / 1024}}
    out = ROOT / "reports" / "level2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    labels = {"actor_numpy": "Actor, NumPy (.npz)", "actor_onnx": "Actor, ONNX Runtime", "actor_torch": "Actor, PyTorch",
              "prior": "Reflex prior (DLS + null space + repulsion)", "shield_pass": "Safety shield, command accepted (3 candidates in one batch)",
              "shield_obstacle": "Safety shield, obstacle 15 cm above the TCP ("
                                 + ("escape" if shield_case["escape"] else f"scale {shield_case['scale']}") + ")",
              "env_step_with_shield": "Full control step incl. physics (24 substeps), features, policy, prior, shield"}
    md = ["| Component (1 CPU thread) | p50 | p99 |", "|---|---|---|"]
    md += [f"| {labels[k]} | {v['p50_ms']:.3f} ms | {v['p99_ms']:.3f} ms |" for k, v in lat.items()]
    md += ["", f"Actor: {params['actor']} parameters (deployed: {params['actor_npz']} weights + normaliser); "
               f"exports match PyTorch to {max(check.values()):.1e}. Control period: 100 ms."]
    (out / "latency.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"saved {npz}, {onnx_path}, {out / 'latency.json'}, {out / 'latency.md'}")


if __name__ == "__main__":
    main()
