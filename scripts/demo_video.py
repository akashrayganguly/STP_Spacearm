"""Demo video (offscreen): the same episode with a surprise obstacle, prior only vs RL residual + shield.

Picks an S2 (obstacles only) episode from reports/level2/eval_S2.json where the prior alone failed with an
obstacle present and RL + shield succeeded (or use --seed), renders both runs and puts them side by side.

  reports/demo.mp4                 side-by-side video (10 fps)
  reports/figures/demo_strip.png   frame strip of both runs

    python scripts/demo_video.py [--seed S] [--scenario S2] [--size 392 296]
"""
from __future__ import annotations

import argparse
import json

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from spacearm.config import ROOT, deep_update, load_config
from spacearm.envs.space_reach_env import SpaceReachEnv
from spacearm.models.distance_net import DistanceNet
from spacearm.rl.ppo_lag import PPOLagAgent
from spacearm.safety.shield import shield_from_config

SCENARIO_OVERRIDES = {"S2": {"env": {"faults": {"enabled": False}, "noise": {"enabled": False}}}, "S4": {}}
STRIP_FRAMES = 6
MIN_RL_STEPS = 60            # the obstacle appears at 1-5 s: a shorter RL episode may finish before it matters
SEPARATOR = 8                # px between the halves (keeps the width divisible by 8 for the codec)
INK, OK, BAD = (11, 11, 11), (16, 120, 60), (198, 40, 40)


class Zero:
    def act(self, obs):
        return np.zeros((1, 7), np.float32)


def pick_seed(scenario: str) -> int:
    res = json.loads((ROOT / "reports" / "level2" / f"eval_{scenario}.json").read_text(encoding="utf-8"))
    prior = {e["seed"]: e for e in res["episodes_detail"]["prior"]}
    rl = {e["seed"]: e for e in res["episodes_detail"]["rl+shield"]}
    # prior stalls (no success, no collision) with an obstacle; RL + shield also met the obstacle and still succeeded
    good = [s for s in prior if prior[s]["obstacle"] and not prior[s]["success"] and not prior[s]["collision"]
            and rl[s]["success"] and rl[s]["obstacle"] and rl[s]["length"] >= MIN_RL_STEPS]
    if not good:
        raise SystemExit("no episode where the prior stalls and RL + shield gets around the obstacle; pass --seed")
    return min(good, key=lambda s: rl[s]["length"])


def run(cfg, seed, policy, shield_net, w, h):
    env = SpaceReachEnv(cfg, difficulty=1.0)
    if shield_net is not None:
        env.set_shield(shield_from_config(env.kin, cfg, shield_net))
    obs, info = env.reset(seed=seed)
    frames, status = [env.render(w, h)], [(info, "")]
    while True:
        obs, _, te, tr, info = env.step(policy.act(obs["actor"]))
        frames.append(env.render(w, h))
        outcome = "SUCCESS" if info["success"] else ("COLLISION" if info["collision"] else ("timeout" if tr else ""))
        status.append((info, outcome))
        if te or tr:
            break
    env.close()
    return frames, status


def annotate(frame, title, info, outcome, t, font, small):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.text((8, 6), title, fill=INK, font=font)
    line = f"t = {t / 10:4.1f} s   distance {info['d_tcp'] * 100:5.1f} cm   clearance {info['clearance'] * 100:5.1f} cm"
    d.text((8, 28), line, fill=INK, font=small)
    if info.get("shield", {}).get("intervened"):
        d.text((8, 46), "shield active", fill=BAD, font=small)
    if outcome:
        d.text((8, img.height - 26), outcome, fill=OK if outcome == "SUCCESS" else BAD, font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenario", default="S2", choices=list(SCENARIO_OVERRIDES))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--policy", default="models/ppo_lag.pt")
    ap.add_argument("--size", type=int, nargs=2, default=[392, 296], metavar=("W", "H"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = deep_update(load_config(args.config), SCENARIO_OVERRIDES[args.scenario])
    seed = args.seed if args.seed is not None else pick_seed(args.scenario)
    agent = PPOLagAgent.load(ROOT / args.policy)
    dnet = DistanceNet.from_config(cfg)
    dnet.load_state_dict(torch.load(ROOT / cfg["env"]["distance_net_path"], map_location="cpu"))
    w, h = args.size
    left = run(cfg, seed, Zero(), None, w, h)
    right = run(cfg, seed, agent, dnet, w, h)
    font, small = ImageFont.load_default(size=16), ImageFont.load_default(size=12)
    n = max(len(left[0]), len(right[0]))
    frames = []
    for k in range(n):
        halves = []
        for (fr, st), title in ((left, "Prior only (reflex)"), (right, "RL residual + safety shield")):
            i = min(k, len(fr) - 1)
            halves.append(annotate(fr[i], title, st[i][0], st[i][1] if i == len(fr) - 1 else "", i, font, small))
        frames.append(np.concatenate([halves[0], np.full((h, SEPARATOR, 3), 255, np.uint8), halves[1]], axis=1))
    out = ROOT / "reports" / "demo.mp4"
    imageio.mimsave(out, frames, fps=10, macro_block_size=8)
    idx = np.linspace(0, n - 1, STRIP_FRAMES).astype(int)
    strip = np.concatenate([frames[i] for i in idx[: STRIP_FRAMES // 2]], axis=1)
    strip2 = np.concatenate([frames[i] for i in idx[STRIP_FRAMES // 2:]], axis=1)
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([strip, strip2], axis=0)).resize(
        (strip.shape[1] // 2, strip.shape[0])).save(fig_dir / "demo_strip.png")
    lo, ro = left[1][-1], right[1][-1]
    print(f"seed {seed} ({args.scenario}): prior {lo[1] or 'running'} after {len(left[0]) - 1} steps "
          f"(final distance {lo[0]['d_tcp'] * 100:.1f} cm) | RL + shield {ro[1]} after {len(right[0]) - 1} steps")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) and {fig_dir / 'demo_strip.png'}")


if __name__ == "__main__":
    main()
