"""PPO-Lagrangian with asymmetric actor-critic (DESIGN §6.3, D11/D12/D14).

* Actor obs_dim -> 64 -> 64 -> act_dim, tanh mean, state-independent log-std (small exploration for residual RL).
  Parameter-shared over an agent dimension: observations (n_envs, n_agents, obs_dim) -> MAPPO-ready.
* Reward and cost critics on the privileged critic observation (90 -> 256 -> 256 -> 1).
* Running mean/std observation normalisation (clip +-10), stored with the agent.
* lambda <- clip(lambda + lambda_lr * (mean episode cost - cost_limit), 0, lambda_max)   (dual ascent)
  advantage = (A_r - lambda * A_c) / (1 + lambda), standardised per minibatch.
* Any episode end cuts the bootstrap (the critic observes t/T, D14); the rollout boundary bootstraps.
* Residual-RL safeguards: critics train alone for actor_warmup_updates, actor lr < critic lr, both decay linearly.
* `train` is resumable (`resume=` checkpoint dict) and time-boxed (`max_minutes=`); `callback(ctx)` is called
  after every update (the training script uses it for periodic evaluation, checkpoints and best-model selection).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from spacearm.rl.vec_env import SubprocVecEnv, SyncVecEnv

OBS_CLIP = 10.0
_EPS = 1e-8


# ------------------------------------------------------------------ building blocks
class RunningMeanStd:
    """Streaming mean/variance (parallel algorithm of Chan et al.)."""

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = float(epsilon)

    def update(self, x) -> None:
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var, self.count = m2 / tot, tot

    def normalize(self, x) -> np.ndarray:
        return np.clip((np.asarray(x) - self.mean) / np.sqrt(self.var + _EPS), -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load(self, s: dict) -> None:
        self.mean, self.var, self.count = np.array(s["mean"]), np.array(s["var"]), float(s["count"])


def compute_gae(rewards, values, next_values, terminated, truncated, gamma: float, lam: float):
    """GAE over (T, N) arrays. Termination stops the bootstrap; any episode end (terminated or truncated)
    cuts the advantage trace. Returns (advantages, returns = advantages + values)."""
    rewards, values, next_values = (np.asarray(a, np.float64) for a in (rewards, values, next_values))
    term = np.asarray(terminated, np.float64)
    done = np.maximum(term, np.asarray(truncated, np.float64))
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1:])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * (1.0 - term[t]) * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + values


class LagrangeMultiplier:
    """Dual ascent on the constraint 'mean episode cost <= limit', projected to [0, max_value]."""

    def __init__(self, init: float, lr: float, max_value: float):
        self.value, self.lr, self.max_value = float(init), float(lr), float(max_value)

    def update(self, mean_episode_cost: float, cost_limit: float) -> float:
        self.value = float(np.clip(self.value + self.lr * (mean_episode_cost - cost_limit), 0.0, self.max_value))
        return self.value


def _mlp(sizes, act=nn.Tanh, out_gain=1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = nn.Linear(a, b)
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, gain=out_gain if last else np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin] if last else [lin, act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy with a tanh-bounded mean (deterministic action) and a state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), log_std_init: float = -1.6):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, act_dim], out_gain=0.01)       # starts near zero residual
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self(obs), self.log_std.exp().expand(obs.shape[0], -1))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, 1], out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PPOLagAgent:
    """Actor + reward/cost critics + observation normalisers. `act` is the deterministic deployed policy."""

    def __init__(self, obs_dim: int, critic_dim: int, act_dim: int, ppo_cfg: dict):
        self.obs_dim, self.critic_dim, self.act_dim = obs_dim, critic_dim, act_dim
        self.ppo_cfg = dict(ppo_cfg)
        self.actor = Actor(obs_dim, act_dim, ppo_cfg["actor_hidden"], ppo_cfg["log_std_init"])
        self.critic_r = Critic(critic_dim, ppo_cfg["critic_hidden"])
        self.critic_c = Critic(critic_dim, ppo_cfg["critic_hidden"])
        self.actor_rms = RunningMeanStd((obs_dim,))
        self.critic_rms = RunningMeanStd((critic_dim,))

    def act(self, obs) -> np.ndarray:
        """Deterministic action for raw actor observations (..., obs_dim) -> (..., act_dim) float32."""
        obs = np.asarray(obs, np.float32)
        with torch.no_grad():
            a = self.actor(torch.from_numpy(self.actor_rms.normalize(obs.reshape(-1, self.obs_dim))))
        return a.numpy().reshape(*obs.shape[:-1], self.act_dim).astype(np.float32)

    def state(self) -> dict:
        return {"dims": (self.obs_dim, self.critic_dim, self.act_dim), "ppo_cfg": self.ppo_cfg,
                "actor": self.actor.state_dict(), "critic_r": self.critic_r.state_dict(),
                "critic_c": self.critic_c.state_dict(), "actor_rms": self.actor_rms.state(),
                "critic_rms": self.critic_rms.state()}

    @classmethod
    def from_state(cls, s: dict) -> "PPOLagAgent":
        agent = cls(*s["dims"], s["ppo_cfg"])
        agent.actor.load_state_dict(s["actor"])
        agent.critic_r.load_state_dict(s["critic_r"])
        agent.critic_c.load_state_dict(s["critic_c"])
        agent.actor_rms.load(s["actor_rms"])
        agent.critic_rms.load(s["critic_rms"])
        return agent

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path)
        return path

    @classmethod
    def load(cls, path) -> "PPOLagAgent":
        # numpy arrays inside -> weights_only=False (torch >= 2.6 defaults to True; lesson 20)
        return cls.from_state(torch.load(Path(path), map_location="cpu", weights_only=False))


# ------------------------------------------------------------------ evaluation
def _run_episode_stats(env, policy, seed):
    obs, info = env.reset(seed=seed)
    ret = cost = 0.0
    n = 0
    while True:
        obs, r, te, tr, info = env.step(policy.act(obs["actor"]))
        ret += r
        cost += info.get("cost", 0.0)
        n += 1
        if te or tr:
            break
    return {"return": ret, "cost": cost, "length": n, "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)), "base_rotation_deg": info.get("base_rotation_deg", np.nan),
            "shield_interventions": info.get("shield_interventions", np.nan)}


def summarize_episodes(eps: list[dict]) -> dict:
    out = {k: float(np.mean([e[k] for e in eps])) for k in ("success", "collision", "cost", "return", "length")}
    succ = [e["length"] for e in eps if e["success"]]
    out["success_length"] = float(np.mean(succ)) if succ else float("nan")
    out["n_episodes"] = len(eps)
    return out


def evaluate(agent, env, n_episodes: int = 20, seed: int = 0) -> dict:
    """Deterministic policy (agent.act) on n fixed-seed episodes (seed, seed + 1, ...) of one env."""
    eps = [_run_episode_stats(env, agent, seed + k) for k in range(int(n_episodes))]
    res = summarize_episodes(eps)
    res["episodes"] = eps
    return res


def evaluate_vec(agent, venv, n_episodes: int, seed: int) -> dict:
    """Parallel deterministic evaluation: reset the vector env with `seed` and collect the first
    n_episodes / n_envs episodes of every env (reproducible for a fixed seed and policy)."""
    per_env = int(np.ceil(n_episodes / venv.n))
    obs = venv.reset(seed=seed)
    venv.pop_episodes()
    counts = np.zeros(venv.n, int)
    eps: list[list[dict]] = [[] for _ in range(venv.n)]
    ret = np.zeros(venv.n)
    cost = np.zeros(venv.n)
    length = np.zeros(venv.n, int)
    while np.any(counts < per_env):
        obs, r, c, te, tr, infos = venv.step(agent.act(obs["actor"]))
        ret += r
        cost += c
        length += 1
        for i in np.flatnonzero(te | tr):
            if counts[i] < per_env:
                eps[i].append({"return": ret[i], "cost": cost[i], "length": int(length[i]),
                               "success": bool(infos[i].get("success", False)),
                               "collision": bool(infos[i].get("collision", False))})
                counts[i] += 1
            ret[i] = cost[i] = 0.0
            length[i] = 0
    venv.pop_episodes()
    flat = [e for env_eps in eps for e in env_eps][: int(n_episodes)]
    return summarize_episodes(flat)


# ------------------------------------------------------------------ training
def _lr_now(base: float, steps: int, total: int) -> float:
    return base * max(0.0, 1.0 - steps / max(1, total))


def train(make_env, cfg: dict, total_steps: int | None = None, n_envs: int | None = None, seed: int = 0,
          vec: str = "sync", curriculum: bool = False, log_fn=print, save_path=None, callback=None,
          resume: dict | None = None, max_minutes: float | None = None, n_steps: int | None = None):
    """Train a PPOLagAgent. Returns (agent, history) with one dict per update.

    make_env: picklable env factory (a class or functools.partial). curriculum (opt-in, the arm task uses it):
    difficulty ramps 0 -> 1 over ppo.curriculum_frac of training. resume: a checkpoint dict from
    ctx["checkpoint"]() of an earlier run. max_minutes: stop after the update that exceeds this wall time.
    callback(ctx) after every update; ctx has agent, update, steps, total_steps, history, lambda, difficulty,
    checkpoint (a function returning the full training state); a truthy return value stops training."""
    p = cfg["ppo"]
    total_steps = int(p["total_steps"] if total_steps is None else total_steps)
    n_envs = int(p["n_envs"] if n_envs is None else n_envs)
    n_steps = int(p["n_steps"] if n_steps is None else n_steps)
    torch.manual_seed(seed)
    venv = (SubprocVecEnv if vec == "subproc" else SyncVecEnv)([make_env] * n_envs)
    obs_shape = venv.observation_space["actor"].shape            # (n_agents, obs_dim)
    n_agents, obs_dim = obs_shape
    critic_dim = venv.observation_space["critic"].shape[0]
    act_dim = venv.action_space.shape[-1]

    agent = PPOLagAgent(obs_dim, critic_dim, act_dim, p)
    lam = LagrangeMultiplier(p["lambda_init"], p["lambda_lr"], p["lambda_max"])
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=p["actor_lr"])
    opt_c = torch.optim.Adam(list(agent.critic_r.parameters()) + list(agent.critic_c.parameters()), lr=p["lr"])
    update, steps, history = 0, 0, []
    if resume is not None:
        agent = PPOLagAgent.from_state(resume["agent"])
        opt_a = torch.optim.Adam(agent.actor.parameters(), lr=p["actor_lr"])
        opt_c = torch.optim.Adam(list(agent.critic_r.parameters()) + list(agent.critic_c.parameters()), lr=p["lr"])
        opt_a.load_state_dict(resume["opt_actor"])
        opt_c.load_state_dict(resume["opt_critic"])
        lam.value = float(resume["lambda"])
        update, steps, history = int(resume["update"]), int(resume["steps"]), list(resume["history"])
        torch.set_rng_state(resume["torch_rng"])

    def checkpoint() -> dict:
        return {"agent": agent.state(), "opt_actor": opt_a.state_dict(), "opt_critic": opt_c.state_dict(),
                "lambda": lam.value, "update": update, "steps": steps, "history": history,
                "torch_rng": torch.get_rng_state(), "total_steps": total_steps, "seed": seed}

    def difficulty_at(s: int) -> float:
        return 1.0 if not curriculum else min(1.0, s / max(1.0, p["curriculum_frac"] * total_steps))

    venv.set_difficulty(difficulty_at(steps))
    obs = venv.reset(seed=seed + 1000 * update)                   # fresh env seeds on resume
    t_start = time.perf_counter()
    batch = n_envs * n_steps
    learner_threads = torch.get_num_threads()     # 1 thread while env workers run (no OpenMP spin-waits), all for updates
    try:
        while steps < total_steps:
            t0 = time.perf_counter()
            d = difficulty_at(steps)
            venv.set_difficulty(d)
            torch.set_num_threads(1)
            # ---------------- rollout (store normalised observations exactly as used for acting)
            buf_a = np.zeros((n_steps, n_envs, n_agents, obs_dim), np.float32)
            buf_c = np.zeros((n_steps, n_envs, critic_dim), np.float32)
            buf_act = np.zeros((n_steps, n_envs, n_agents, act_dim), np.float32)
            buf_logp = np.zeros((n_steps, n_envs), np.float32)
            buf_r, buf_cost, buf_vr, buf_vc = (np.zeros((n_steps, n_envs)) for _ in range(4))
            buf_term, buf_trunc = np.zeros((n_steps, n_envs)), np.zeros((n_steps, n_envs))
            for t in range(n_steps):
                agent.actor_rms.update(obs["actor"].reshape(-1, obs_dim))
                agent.critic_rms.update(obs["critic"])
                oa = agent.actor_rms.normalize(obs["actor"].reshape(-1, obs_dim))
                oc = agent.critic_rms.normalize(obs["critic"])
                with torch.no_grad():
                    dist = agent.actor.dist(torch.from_numpy(oa))
                    a = dist.sample()
                    logp = dist.log_prob(a).sum(-1).reshape(n_envs, n_agents).sum(-1)
                    oc_t = torch.from_numpy(oc)
                    vr, vc = agent.critic_r(oc_t), agent.critic_c(oc_t)
                a_np = a.numpy().reshape(n_envs, n_agents, act_dim)
                buf_a[t], buf_c[t], buf_act[t], buf_logp[t] = oa.reshape(n_envs, n_agents, obs_dim), oc, a_np, logp.numpy()
                buf_vr[t], buf_vc[t] = vr.numpy(), vc.numpy()
                obs, r, c, te, tr, _ = venv.step(np.clip(a_np, -1.0, 1.0))
                buf_r[t], buf_cost[t] = r, c
                # any episode end cuts the bootstrap (the critic observes t/T, D14)
                buf_term[t] = te | tr
                buf_trunc[t] = tr
            with torch.no_grad():
                oc_last = torch.from_numpy(agent.critic_rms.normalize(obs["critic"]))
                last_vr, last_vc = agent.critic_r(oc_last).numpy(), agent.critic_c(oc_last).numpy()
            next_vr = np.concatenate([buf_vr[1:], last_vr[None]])
            next_vc = np.concatenate([buf_vc[1:], last_vc[None]])
            adv_r, ret_r = compute_gae(buf_r, buf_vr, next_vr, buf_term, buf_trunc, p["gamma"], p["gae_lambda"])
            adv_c, ret_c = compute_gae(buf_cost, buf_vc, next_vc, buf_term, buf_trunc, p["gamma"], p["gae_lambda"])
            steps += batch
            update += 1

            # ---------------- dual ascent on the episode cost
            eps = venv.pop_episodes()
            ep_stats = summarize_episodes(eps) if eps else None
            if ep_stats is not None:
                lam.update(ep_stats["cost"], p["cost_limit"])

            # ---------------- PPO update
            torch.set_num_threads(learner_threads)
            lr_a, lr_c = _lr_now(p["actor_lr"], steps, total_steps), _lr_now(p["lr"], steps, total_steps)
            for g in opt_a.param_groups:
                g["lr"] = lr_a
            for g in opt_c.param_groups:
                g["lr"] = lr_c
            train_actor = update > p["actor_warmup_updates"]
            B = n_steps * n_envs
            fa = torch.from_numpy(buf_a.reshape(B, n_agents, obs_dim))
            fc = torch.from_numpy(buf_c.reshape(B, critic_dim))
            fact = torch.from_numpy(buf_act.reshape(B, n_agents, act_dim))
            flogp = torch.from_numpy(buf_logp.reshape(B))
            fadv = torch.from_numpy(((adv_r - lam.value * adv_c) / (1.0 + lam.value)).reshape(B)).float()
            fret_r = torch.from_numpy(ret_r.reshape(B)).float()
            fret_c = torch.from_numpy(ret_c.reshape(B)).float()
            mb = max(1, B // p["minibatches"])
            kls, clipfracs, v_losses, c_losses = [], [], [], []
            stop = False
            for _ in range(p["epochs"]):
                perm = torch.randperm(B)
                for i in range(0, B, mb):
                    idx = perm[i:i + mb]
                    v_loss = ((agent.critic_r(fc[idx]) - fret_r[idx]) ** 2).mean()
                    c_loss = ((agent.critic_c(fc[idx]) - fret_c[idx]) ** 2).mean()
                    opt_c.zero_grad(set_to_none=True)
                    (p["vf_coef"] * (v_loss + c_loss)).backward()
                    nn.utils.clip_grad_norm_(list(agent.critic_r.parameters()) + list(agent.critic_c.parameters()),
                                             p["max_grad_norm"])
                    opt_c.step()
                    v_losses.append(v_loss.item())
                    c_losses.append(c_loss.item())
                    if not train_actor:
                        continue
                    o = fa[idx].reshape(-1, obs_dim)
                    logp = agent.actor.dist(o).log_prob(fact[idx].reshape(-1, act_dim)).sum(-1)
                    logp = logp.reshape(-1, n_agents).sum(-1)
                    ratio = torch.exp(logp - flogp[idx])
                    adv = fadv[idx]
                    adv = (adv - adv.mean()) / (adv.std() + _EPS) if len(adv) > 1 else adv
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - p["clip"], 1 + p["clip"]) * adv).mean()
                    opt_a.zero_grad(set_to_none=True)
                    pg.backward()
                    nn.utils.clip_grad_norm_(agent.actor.parameters(), p["max_grad_norm"])
                    opt_a.step()
                    with torch.no_grad():
                        log_ratio = logp - flogp[idx]
                        kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    kls.append(kl)
                    clipfracs.append(((ratio - 1).abs() > p["clip"]).float().mean().item())
                    if kl > p["target_kl"]:
                        stop = True
                        break
                if stop:
                    break

            dt = time.perf_counter() - t0
            row = {"update": update, "steps": steps, "difficulty": d, "lambda": lam.value,
                   "return": np.nan, "cost": np.nan, "success": np.nan, "collision": np.nan, "length": np.nan,
                   "episodes": len(eps), "kl": float(np.mean(kls)) if kls else 0.0,
                   "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                   "v_loss": float(np.mean(v_losses)), "c_loss": float(np.mean(c_losses)),
                   "std": float(agent.actor.log_std.exp().mean().item()), "lr_actor": lr_a if train_actor else 0.0,
                   "fps": batch / dt, "minutes": (time.perf_counter() - t_start) / 60}
            if ep_stats is not None:
                row.update({k: ep_stats[k] for k in ("return", "cost", "success", "collision", "length")})
            history.append(row)
            if log_fn is not None:
                log_fn(f"upd {update:5d} steps {steps:8d} d {d:.2f} lam {lam.value:5.2f} | ret {row['return']:7.2f} "
                       f"cost {row['cost']:5.2f} succ {row['success']:5.2f} coll {row['collision']:5.2f} | "
                       f"kl {row['kl']:.4f} std {row['std']:.3f} | {row['fps']:5.0f} sps")
            if callback is not None and callback({"agent": agent, "update": update, "steps": steps,
                                                  "total_steps": total_steps, "history": history, "lambda": lam.value,
                                                  "difficulty": d, "checkpoint": checkpoint}):
                break
            if max_minutes is not None and (time.perf_counter() - t_start) / 60 >= max_minutes:
                break
    finally:
        torch.set_num_threads(learner_threads)
        venv.close()
    if save_path is not None:
        agent.save(save_path)
    return agent, history
