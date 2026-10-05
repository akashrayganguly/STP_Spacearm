"""Vectorised environments with same-step auto-reset (DESIGN §6.3, lesson 17).

step(actions (N, 1, A)) -> (obs, reward (N,), cost (N,), terminated (N,), truncated (N,), infos):
when an episode ends, the reward/cost/flags of its final step are kept, its statistics are recorded
(pop_episodes), the env is reset and the returned observation is the NEW episode's first one; the final
observation of the ended episode is in infos[i]["final_obs"] (for value bootstrapping at truncation).
Gymnasium's own vector envs changed their auto-reset semantics in 1.x, so we use our own.
SubprocVecEnv uses the "spawn" start method (Windows-safe); env factories must be picklable
(a class, or functools.partial(SpaceReachEnv, cfg)).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _stack(obs_list: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


class _Episode:
    """Running statistics of the current episode of one env."""

    def __init__(self):
        self.ret, self.cost, self.length = 0.0, 0.0, 0

    def add(self, r: float, c: float) -> None:
        self.ret += r
        self.cost += c
        self.length += 1


def _step_autoreset(env, action, ep: _Episode):
    """Step one env; on episode end record its statistics and reset (same-step auto-reset)."""
    obs, r, te, tr, info = env.step(action)
    c = float(info.get("cost", 0.0))
    ep.add(r, c)
    done_ep = None
    if te or tr:
        done_ep = {"return": ep.ret, "cost": ep.cost, "length": ep.length, "success": bool(info.get("success", False)),
                   "collision": bool(info.get("collision", False)), "truncated": bool(tr)}
        info = dict(info, final_obs=obs)
        obs, _ = env.reset()
        ep.__init__()
    return obs, float(r), c, bool(te), bool(tr), info, done_ep


class SyncVecEnv:
    """All envs in this process (simple, debuggable)."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)
        self._eps = [_Episode() for _ in range(self.n)]
        self._done: list[dict] = []
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed: int | None = None) -> dict:
        obs = [env.reset(seed=None if seed is None else seed + i)[0] for i, env in enumerate(self.envs)]
        self._eps = [_Episode() for _ in range(self.n)]
        return _stack(obs)

    def step(self, actions):
        out = [_step_autoreset(env, a, ep) for env, a, ep in zip(self.envs, actions, self._eps)]
        obs, r, c, te, tr, infos, done = zip(*out)
        self._done += [d for d in done if d is not None]
        return _stack(list(obs)), np.array(r), np.array(c), np.array(te), np.array(tr), list(infos)

    def set_difficulty(self, d: float) -> None:
        for env in self.envs:
            env.set_difficulty(d)

    def pop_episodes(self) -> list[dict]:
        done, self._done = self._done, []
        return done

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _worker(remote, parent_remote, env_fn) -> None:
    parent_remote.close()
    import torch

    torch.set_num_threads(1)                       # one thread per env worker (lesson 18)
    env, ep = env_fn(), _Episode()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(_step_autoreset(env, data, ep))
            elif cmd == "reset":
                ep.__init__()
                remote.send(env.reset(seed=data)[0])
            elif cmd == "set_difficulty":
                env.set_difficulty(data)
                remote.send(None)
            elif cmd == "spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()
        remote.close()


class SubprocVecEnv:
    """One env per worker process ("spawn" start method)."""

    def __init__(self, env_fns):
        ctx = mp.get_context("spawn")
        self.n = len(env_fns)
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.procs = []
        for work, remote, fn in zip(work_remotes, self.remotes, env_fns):
            proc = ctx.Process(target=_worker, args=(work, remote, fn), daemon=True)
            proc.start()
            self.procs.append(proc)
            work.close()
        self.remotes[0].send(("spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self._done: list[dict] = []
        self.closed = False

    def reset(self, seed: int | None = None) -> dict:
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", None if seed is None else seed + i))
        return _stack([remote.recv() for remote in self.remotes])

    def step(self, actions):
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", a))
        obs, r, c, te, tr, infos, done = zip(*[remote.recv() for remote in self.remotes])
        self._done += [d for d in done if d is not None]
        return _stack(list(obs)), np.array(r), np.array(c), np.array(te), np.array(tr), list(infos)

    def set_difficulty(self, d: float) -> None:
        for remote in self.remotes:
            remote.send(("set_difficulty", d))
        for remote in self.remotes:
            remote.recv()

    def pop_episodes(self) -> list[dict]:
        done, self._done = self._done, []
        return done

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
        self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
