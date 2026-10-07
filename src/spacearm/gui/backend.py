"""GUI backend (no Gradio): Level-1 pose and planner queries, Level-2 missions with recording and video.

Every number the app shows is computed with the same code the evaluations use:
  true clearances       PyBullet closest points over the 30 body + 10 self link pairs (SpaceRobotSim)
  predicted clearances  the trained DistanceNet (models/distance_net.pt)
  planning              Level1Planner (TrajNet + refine + IK polish) and PyBullet verification
  missions              SpaceReachEnv.reset(options=...) + reflex prior, PPO-Lagrangian actor, safety shield
"""
from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import plotly.graph_objects as go
import pybullet as p
import torch
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots

from spacearm.config import ROOT, deep_update, load_config
from spacearm.envs.space_reach_env import SpaceReachEnv
from spacearm.gui import scene
from spacearm.kinematics import ArmKinematics
from spacearm.models.distance_net import DistanceNet
from spacearm.models.traj_net import TrajNet
from spacearm.planner import Level1Planner, ik_baseline, plan_and_verify, verify_trajectory
from spacearm.rl.ppo_lag import PPOLagAgent
from spacearm.robot_model import panel_centres
from spacearm.safety.shield import shield_from_config
from spacearm.sim import SpaceRobotSim

LOCK = threading.RLock()            # PyBullet clients and torch modules are shared: one request at a time
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REFERENCE = "#898781"               # baselines / references are drawn in neutral grey
CRITICAL, MARGIN_INK = "#d03b3b", "#898781"
LINK_NAMES = {"bus": "bus", "panel_left": "left solar panel", "panel_right": "right solar panel",
              "payload": "payload box", "pedestal": "pedestal", **{f"link{i}": f"arm link {i}" for i in range(1, 8)}}
CONTROLLERS = {"Reflex only": (False, False), "Reflex + shield": (False, True), "RL": (True, False),
               "RL + shield": (True, True)}
# Scenario configs exactly as in scripts/eval_level2.py, so report episodes (seeds 10000+) can be replayed.
SCENARIOS = {"S1 nominal": (0.0, {}),
             "S2 obstacles only": (1.0, {"env": {"faults": {"enabled": False}, "noise": {"enabled": False}}}),
             "S3 faults + noise only": (1.0, {"env": {"obstacle": {"prob": 0.0}}}),
             "S4 everything": (1.0, {})}


# ------------------------------------------------------------------ models (loaded once)
@dataclass
class Models:
    cfg: dict
    kin: ArmKinematics                       # float64: exact geometry for display
    sim: SpaceRobotSim                       # body-frame queries (base fixed at the origin)
    dnet: DistanceNet
    planner: Level1Planner
    agent: PPOLagAgent | None
    index_to_name: dict
    envs: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def models() -> Models:
    with LOCK:
        torch.set_num_threads(1)
        cfg = load_config()
        mdir = ROOT / cfg["paths"]["models_dir"]
        kin32 = ArmKinematics(cfg)
        dnet = DistanceNet.from_config(cfg)
        dnet.load_state_dict(torch.load(mdir / "distance_net.pt", map_location="cpu"))
        dnet.eval().requires_grad_(False)
        tnet = TrajNet.from_config(cfg, kin32.lower, kin32.upper)
        tnet.load_state_dict(torch.load(mdir / "traj_net.pt", map_location="cpu"))
        sim = SpaceRobotSim(cfg)
        agent = PPOLagAgent.load(mdir / "ppo_lag.pt") if (mdir / "ppo_lag.pt").exists() else None
        return Models(cfg=cfg, kin=ArmKinematics(cfg, dtype=torch.float64), sim=sim, dnet=dnet,
                      planner=Level1Planner(kin32, dnet, tnet, cfg), agent=agent,
                      index_to_name={v: k for k, v in sim.link_index.items()})


def joint_limits_deg() -> tuple[list[float], list[float]]:
    arm = models().cfg["robot"]["arm"]
    return list(arm["lower_deg"]), list(arm["upper_deg"])


def tcp_of(q) -> np.ndarray:
    return scene.arm_geometry(models().kin, q)[3]


def tip_path(Q) -> np.ndarray:
    m = models()
    with torch.no_grad():
        return m.kin.forward(torch.as_tensor(np.asarray(Q, float), dtype=m.kin.dtype))[1][:, -1].numpy()


# ------------------------------------------------------------------ Level 1: one pose
def _closest(m: Models, pairs, max_dist: float):
    best = (max_dist, None)
    for a, b in pairs:
        for pt in p.getClosestPoints(m.sim.robot, m.sim.robot, max_dist, linkIndexA=a, linkIndexB=b,
                                     physicsClientId=m.sim.cid):
            if pt[8] < best[0]:
                best = (pt[8], {"links": (LINK_NAMES[m.index_to_name[a]], LINK_NAMES[m.index_to_name[b]]),
                                "p_a": np.array(pt[5]), "p_b": np.array(pt[6])})
    return best


def pose_report(q_deg) -> dict:
    """True (PyBullet) and predicted (DistanceNet) clearances of one pose, with the closest link pairs."""
    m = models()
    q = np.radians(np.asarray(q_deg, float))
    max_dist = m.cfg["sim"]["max_query_dist"]
    with LOCK:
        m.sim.set_q(q)
        p.performCollisionDetection(physicsClientId=m.sim.cid)
        d_body, body = _closest(m, m.sim.pairs_body, max_dist)
        d_self, selfp = _closest(m, m.sim.pairs_self, max_dist)
        with torch.no_grad():
            qt = torch.as_tensor(q, dtype=torch.float32)
            pred = m.dnet(qt).numpy()
            clear = float(m.dnet.clearance(qt))
    true_min = min(d_body, d_self)
    status = "COLLISION" if true_min <= 0 else ("CLOSE (inside the 5 cm planning margin)" if true_min < 0.05 else "SAFE")
    return {"q": q, "d_body": d_body, "d_self": d_self, "body": body, "self": selfp, "pred_body": float(pred[0]),
            "pred_self": float(pred[1]), "pred_clearance": clear, "tcp": tcp_of(q), "status": status,
            "clipped": max_dist}


def _fmt_cm(d, clip):
    return f"≥ {100 * clip:.0f} cm" if d >= clip - 1e-9 else f"{100 * d:+.1f} cm"


def pose_figure(rep: dict, height: int = 600) -> go.Figure:
    m = models()
    extra = []
    for key, color, label in (("body", scene.COLORS["body_pair"], "d_body"), ("self", scene.COLORS["self_pair"], "d_self")):
        if rep[key] is not None:
            extra.append(scene.segment_trace(rep[key]["p_a"], rep[key]["p_b"], color,
                                             f"{label} {100 * rep['d_' + key]:+.1f} cm"))
    fig = scene.robot_figure(m.cfg, m.kin, rep["q"], extra=extra, height=height, uirevision="pose")
    clip = rep["clipped"]
    txt = (f"<b>{rep['status']}</b><br>"
           f"d_body  true {_fmt_cm(rep['d_body'], clip)} · DistanceNet {100 * rep['pred_body']:+.1f} cm<br>"
           f"d_self   true {_fmt_cm(rep['d_self'], clip)} · DistanceNet {100 * rep['pred_self']:+.1f} cm<br>"
           f"tool tip  x {rep['tcp'][0]:+.2f}  y {rep['tcp'][1]:+.2f}  z {rep['tcp'][2]:+.2f} m")
    fig.add_annotation(text=txt, x=0.99, y=0.98, xref="paper", yref="paper", xanchor="right", yanchor="top",
                       align="left", showarrow=False, font={"size": 13, "color": scene.INK},
                       bgcolor="rgba(252,252,251,0.88)", bordercolor="#e1e0d9", borderwidth=1, borderpad=8)
    return fig


def random_pose(kind: str, seed: int) -> list[float]:
    """A random pose (degrees) that is 'safe' (> 5 cm clear), 'close' (0-5 cm) or 'colliding'."""
    m = models()
    rng = np.random.default_rng(int(seed))
    frac = m.cfg["data"]["limit_frac"]
    mid, half = (m.sim.lower + m.sim.upper) / 2, (m.sim.upper - m.sim.lower) / 2
    with LOCK:
        for _ in range(5000):
            q = rng.uniform(mid - frac * half, mid + frac * half)
            m.sim.set_q(q)
            d = min(m.sim.min_distances())
            if (kind == "safe" and d > 0.05) or (kind == "close" and 0.0 < d < 0.05) or (kind == "colliding" and d < 0):
                return np.round(np.degrees(q), 1).tolist()
    raise RuntimeError(f"no {kind} pose found")


# ------------------------------------------------------------------ Level 1: planner
def shoulder_and_reach():
    r = models().cfg["robot"]
    shoulder = np.array(r["mount_xyz"]) + [0.0, 0.0, r["pedestal"]["length"] + r["arm"]["link_lengths"][0]]
    return shoulder, float(sum(r["arm"]["link_lengths"][1:6]) + r["arm"]["tool_offset"])


def check_target(target) -> tuple[bool, str]:
    """Reachable (within the arm's reach from the shoulder) and outside the spacecraft parts?"""
    m = models()
    t = np.asarray(target, float)
    shoulder, reach = shoulder_and_reach()
    dist = float(np.linalg.norm(t - shoulder))
    if dist > reach:
        return False, f"Target is {dist:.2f} m from the shoulder; the arm reaches at most {reach:.2f} m."
    r = m.cfg["robot"]
    boxes = [([0, 0, 0], r["bus"]["size"])] + [(c, r["panels"]["size"]) for c in panel_centres(m.cfg).values()]
    boxes += [(pl["xyz"], pl["size"]) for pl in r["payloads"]]
    for c, s in boxes:
        if np.all(np.abs(t - np.asarray(c)) <= np.asarray(s) / 2 + 0.02):
            return False, "Target is inside (or within 2 cm of) the spacecraft."
    note = "" if dist <= 0.85 * reach else (f" Note: {dist:.2f} m is beyond 85 % of the reach ({0.85 * reach:.2f} m); "
                                            "Level 2 never trains on such targets.")
    return True, f"Target OK ({dist:.2f} m from the shoulder).{note}"


def clearance_along(Q) -> np.ndarray:
    m = models()
    out = np.empty(len(Q))
    with LOCK:
        for k, q in enumerate(Q):
            m.sim.set_q(q)
            out[k] = min(m.sim.min_distances())
    return out


STAGES = [("TrajNet alone", 0, False), ("+ refine", None, False), ("+ refine + IK polish", None, True)]


def plan_report(q_start_deg, target, baseline: bool = True) -> dict:
    """Run every planner stage on one query and verify each in PyBullet."""
    m = models()
    q = np.radians(np.asarray(q_start_deg, float))
    t = np.asarray(target, float)
    tol = m.cfg["traj"]["success_tol"]
    rows = []
    with LOCK:
        if baseline:
            t0 = time.perf_counter()
            Q = ik_baseline(m.sim, q, t, m.cfg["traj"]["t_verify"])
            dt = time.perf_counter() - t0
            v = verify_trajectory(m.sim, Q, t, tol)
            rows.append({"name": "Baseline: IK + straight line", "Q": Q, "cp": None, "time": dt, **v})
        for name, steps, polish in STAGES:
            t0 = time.perf_counter()
            Q, info = m.planner.plan(q, t, refine_steps=steps, polish=polish)
            dt = time.perf_counter() - t0
            v = verify_trajectory(m.sim, Q, t, tol)
            rows.append({"name": name, "Q": Q, "cp": info["control_points"], "time": dt, **v})
        t0 = time.perf_counter()
        Q, info = plan_and_verify(m.planner, m.sim, q, t)
        rows.append({"name": "Full planner (verified" + (", retried)" if info["retried"] else ")"), "Q": Q,
                     "cp": info["control_points"], "time": time.perf_counter() - t0,
                     **{k: info[k] for k in ("success", "collision_free", "min_dist", "reach_err", "within_limits")}})
    for r in rows:
        r["clearance"] = clearance_along(r["Q"])
        r["tip"] = tip_path(r["Q"])
    return {"q_start": q, "target": t, "rows": rows}


def stage_color(name: str, k: int) -> str:
    return REFERENCE if name.startswith("Baseline") else SERIES[(k - 1) % len(SERIES)]


def plan_table(rep: dict) -> list[list]:
    out = []
    for r in rep["rows"]:
        verdict = "✅ success" if r["success"] else ("❌ collides" if not r["collision_free"] else
                                                     ("❌ misses target" if r["reach_err"] >= 0.02 else "❌ limits"))
        out.append([r["name"], verdict, f"{100 * r['min_dist']:+.1f} cm", f"{100 * r['reach_err']:.2f} cm",
                    f"{1000 * r['time']:.0f} ms"])
    return out


def plan_figure(rep: dict, height: int = 600) -> go.Figure:
    m = models()
    final = rep["rows"][-1]
    traces = [scene.spacecraft_trace(m.cfg),
              scene.arm_trace(m.kin, rep["q_start"], opacity=0.35, name="start pose", ghost=True)]
    for s in (0.33, 0.66):
        traces.append(scene.arm_trace(m.kin, final["Q"][int(s * (len(final["Q"]) - 1))], opacity=0.18,
                                      name="along the path", ghost=True))
    traces.append(scene.arm_trace(m.kin, final["Q"][-1], name="goal pose (full planner)"))
    offset = 0 if rep["rows"][0]["name"].startswith("Baseline") else 1
    for k, r in enumerate(rep["rows"]):
        traces.append(scene.line_trace(r["tip"], stage_color(r["name"], k + offset), f"tip path: {r['name']}", width=6))
    traces.append(scene.point_trace(rep["target"], scene.COLORS["target"], "target", size=8))
    fig = go.Figure(traces, layout=scene.layout(height=height, uirevision="plan"))
    fig.update_layout(legend={"x": 0.99, "xanchor": "right", "y": 0.99, "yanchor": "top"})
    return fig


def _chart_layout(fig, title, ytitle, xtitle, height=400):
    fig.update_layout(title={"text": title, "x": 0.01, "font": {"size": 14, "color": scene.INK}}, height=height,
                      paper_bgcolor=scene.SURFACE, plot_bgcolor=scene.SURFACE, margin={"l": 60, "r": 20, "t": 46, "b": 120},
                      legend={"orientation": "h", "y": -0.3, "yanchor": "top", "font": {"size": 11}},
                      hovermode="x unified", font={"color": scene.INK})
    fig.update_xaxes(title=xtitle, gridcolor=scene.GRID, zeroline=False, linecolor="#c3c2b7")
    fig.update_yaxes(title=ytitle, gridcolor=scene.GRID, zeroline=False, linecolor="#c3c2b7")
    return fig


def plan_clearance_chart(rep: dict) -> go.Figure:
    fig = go.Figure()
    offset = 0 if rep["rows"][0]["name"].startswith("Baseline") else 1
    s = np.linspace(0, 1, len(rep["rows"][0]["clearance"]))
    for k, r in enumerate(rep["rows"]):
        fig.add_scatter(x=s, y=100 * r["clearance"], mode="lines", name=r["name"],
                        line={"color": stage_color(r["name"], k + offset), "width": 2})
    fig.add_hline(y=0, line={"color": CRITICAL, "width": 1}, annotation_text="contact", annotation_position="bottom right")
    fig.add_hline(y=5, line={"color": MARGIN_INK, "width": 1}, annotation_text="5 cm planning margin",
                  annotation_position="top right")
    return _chart_layout(fig, "True clearance along each path (PyBullet, 100 points)", "clearance (cm)",
                         "fraction of the path s (0 = start, 1 = goal)")


def plan_joint_chart(rep: dict) -> go.Figure:
    """Joint angles along the final path, with its Bezier control points (dots at s = k/7)."""
    final = rep["rows"][-1]
    s = np.linspace(0, 1, len(final["Q"]))
    cp = final["cp"]
    fig = go.Figure()
    for j in range(7):
        col = SERIES[j]
        fig.add_scatter(x=s, y=np.degrees(final["Q"][:, j]), mode="lines", name=f"joint {j + 1}",
                        line={"color": col, "width": 2}, legendgroup=str(j))
        if cp is not None:
            fig.add_scatter(x=np.arange(len(cp)) / (len(cp) - 1), y=np.degrees(cp[:, j]), mode="markers",
                            marker={"color": col, "size": 9, "line": {"color": scene.SURFACE, "width": 2}},
                            name=f"control points {j + 1}", legendgroup=str(j), showlegend=False)
    return _chart_layout(fig, "Final path: joint angles (lines) and their 8 Bezier control points (dots)",
                         "angle (deg)", "fraction of the path s")


# ------------------------------------------------------------------ Level 2: missions
@dataclass
class MissionSpec:
    seed: int = 0
    start_deg: list | None = None              # None = sampled from the seed (collision-free)
    target: list | None = None                 # None = sampled from the seed (reachable)
    source: str = "manual"                     # "manual" = surprises below; otherwise a key of SCENARIOS
    obstacle: str = "drifting"                 # none | static | drifting
    radius_cm: float = 8.0
    appear_s: float = 2.5
    drift_cm_s: float = 2.0
    obstacle_center: list | None = None        # None = between the hand and the target (training rule)
    bias_deg: float = 0.0                      # per joint, uniform in [-bias, bias] (from the seed)
    motor_min: float = 1.0                     # per-joint motor strength, uniform in [motor_min, 1]
    slip_s: float | None = None                # one +-3 deg encoder slip on a random joint at this time
    noise: float = 0.0                         # x training noise (encoder 0.1 deg, vision 5 mm, gyro 0.002 rad/s)


def mission_options(spec: MissionSpec) -> tuple[dict, float, dict]:
    """(reset options, difficulty, config overrides) for a mission spec."""
    opts = {}
    if spec.start_deg is not None:
        opts["q_start"] = np.radians(np.asarray(spec.start_deg, float))
    if spec.target is not None:
        opts["target"] = np.asarray(spec.target, float)
    if spec.source in SCENARIOS:
        d, over = SCENARIOS[spec.source]
        return opts, d, over
    rng = np.random.default_rng(int(spec.seed) + 7919)
    if spec.obstacle == "none":
        opts["obstacle"] = None
    else:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        vel = direction * spec.drift_cm_s / 100 if spec.obstacle == "drifting" else np.zeros(3)
        opts["obstacle"] = {"time": spec.appear_s, "radius": spec.radius_cm / 100, "velocity": vel,
                            "center": spec.obstacle_center}
    joint, sign = int(rng.integers(0, 7)), float(rng.choice([-1.0, 1.0]))
    opts["faults"] = {"bias_deg": rng.uniform(-spec.bias_deg, spec.bias_deg, 7),
                      "gains": rng.uniform(spec.motor_min, 1.0, 7),
                      "slip_time": spec.slip_s, "slip_joint": joint, "slip_deg": 3.0 * sign}
    n = models().cfg["env"]["noise"]
    opts["noise"] = {"encoder_deg": n["encoder_deg"] * spec.noise, "vision_m": n["vision_m"] * spec.noise,
                     "gyro": n["gyro"] * spec.noise}
    return opts, 1.0, {}


def _env(over: dict) -> SpaceReachEnv:
    m = models()
    key = repr(over)
    if key not in m.envs:
        cfg = deep_update(m.cfg, over)
        env = SpaceReachEnv(cfg, difficulty=1.0)
        env.shield_obj = shield_from_config(env.kin, cfg, env.dnet)
        m.envs[key] = env
    return m.envs[key]


def _quat_to_R(qxyzw):
    return np.array(p.getMatrixFromQuaternion(list(qxyzw))).reshape(3, 3)


def run_mission(controller: str, spec: MissionSpec, render: bool = True, size=(448, 336)) -> dict:
    """One Level-2 episode with full recording (and rendered frames)."""
    use_rl, shield = CONTROLLERS[controller]
    m = models()
    if use_rl and m.agent is None:
        raise RuntimeError("models/ppo_lag.pt is missing")
    opts, d, over = mission_options(spec)
    with LOCK:
        env = _env(over)
        env.set_difficulty(d)
        env.set_shield(env.shield_obj if shield else None)
        obs, info = env.reset(seed=int(spec.seed), options=opts)
        rec = {k: [] for k in ("t", "q", "base_p", "base_q", "tcp", "d_tcp", "d_body", "d_self", "d_obs",
                               "obs_center", "obs_radius", "shield", "cmd", "prior")}
        frames = []
        sampled = {"bias_deg": np.degrees(env.bias).round(2).tolist(), "gains": env.gains.round(3).tolist(),
                   "slip_s": env.slip_step * env.dt if env.slip_step > 0 else None,
                   "obstacle_s": env.spawn_step * env.dt if env.spawn_step > 0 else None,
                   "noise_enc_deg": float(np.degrees(env.sig_enc)), "target": env.target_w.tolist(),
                   "start_deg": np.degrees(env.sim.get_q()).round(1).tolist()}

        def record(info, cmd, intervened):
            pos, orn = env.sim.base_pose()
            rec["t"].append(env.t * env.dt)
            rec["q"].append(env.sim.get_q())
            rec["base_p"].append(pos)
            rec["base_q"].append(orn)
            rec["tcp"].append(env.sim.tcp_world())
            for k in ("d_tcp", "d_body", "d_self", "d_obs"):
                rec[k].append(info[k])
            rec["obs_center"].append(env.obstacle["center"].copy() if env.obstacle is not None else None)
            rec["obs_radius"].append(env.obstacle["radius"] if env.obstacle is not None else None)
            rec["shield"].append(bool(intervened))
            rec["cmd"].append(np.asarray(cmd, float))
            rec["prior"].append(np.asarray(info["prior"], float))

        record(info, np.zeros(7), False)
        if render:
            frames.append(_annotate(env.render(*size), controller, info, False, ""))
        outcome = "running"
        while True:
            a = m.agent.act(obs["actor"]) if use_rl else np.zeros((1, 7), np.float32)
            obs, _, te, tr, info = env.step(a)
            iv = bool(info.get("shield", {}).get("intervened", False))
            record(info, info["cmd"], iv)
            if te or tr:
                outcome = "SUCCESS" if info["success"] else ("COLLISION" if info["collision"] else "TIME-OUT")
            if render:
                frames.append(_annotate(env.render(*size), controller, info, iv, "" if outcome == "running" else outcome))
            if te or tr:
                break
        out = {k: (np.asarray(v, float) if k not in ("obs_center", "obs_radius") else v) for k, v in rec.items()}
        seen = [i for i, c in enumerate(rec["obs_center"]) if c is not None]
        # The training rule needs a free spot between the hand and the target; without one there is no ball.
        sampled["obstacle_seen_s"] = rec["t"][seen[0]] if seen else None
        out.update(controller=controller, outcome=outcome, frames=frames, target=env.target_w.copy(),
                   sampled=sampled, interventions=int(np.sum(out["shield"])), dt=env.dt,
                   base_rotation_deg=info["base_rotation_deg"])
        out["min_clearance"] = float(np.min(np.minimum(np.minimum(out["d_body"], out["d_self"]), out["d_obs"])))
        return out


_FONTS = {}


def _font(size):
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.load_default(size=size)
        except TypeError:                           # Pillow < 10.1
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _annotate(frame, title, info, shield_on, outcome):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.text((8, 6), title, fill=(11, 11, 11), font=_font(16))
    d.text((8, 28), f"t = {info['t'] / 10:4.1f} s   to target {info['d_tcp'] * 100:5.1f} cm   "
                    f"clearance {info['clearance'] * 100:5.1f} cm", fill=(11, 11, 11), font=_font(12))
    if shield_on:
        d.text((8, 46), "shield active", fill=(198, 40, 40), font=_font(12))
    if outcome:
        col = (16, 120, 60) if outcome == "SUCCESS" else (198, 40, 40)
        d.text((8, img.height - 26), outcome, fill=col, font=_font(16))
    return np.asarray(img)


def save_video(frames, name: str = "mission", fps: int = 10) -> str:
    """WebM (VP9): plays in Chrome, Edge, Firefox and every Chromium build (H.264 is missing from some)."""
    path = Path(tempfile.mkdtemp(prefix="spacearm_gui_")) / f"{name}.webm"
    imageio.mimsave(path, frames, fps=fps, codec="libvpx-vp9", macro_block_size=8,
                    ffmpeg_params=["-deadline", "realtime", "-cpu-used", "8", "-b:v", "1500k"])
    return str(path)


def grid_video(traces: list[dict], fps: int = 10) -> str:
    """2 x 2 grid of the four controllers' frames (shorter runs hold their last frame)."""
    n = max(len(tr["frames"]) for tr in traces)
    h, w = traces[0]["frames"][0].shape[:2]
    gap = 8
    frames = []
    for k in range(n):
        tiles = [tr["frames"][min(k, len(tr["frames"]) - 1)] for tr in traces]
        top = np.concatenate([tiles[0], np.full((h, gap, 3), 255, np.uint8), tiles[1]], 1)
        bottom = np.concatenate([tiles[2], np.full((h, gap, 3), 255, np.uint8), tiles[3]], 1)
        frames.append(np.concatenate([top, np.full((gap, top.shape[1], 3), 255, np.uint8), bottom], 0))
    return save_video(frames, "compare", fps)


def mission_chart(traces: list[dict]) -> go.Figure:
    """Distance to target (top) and true clearances (bottom) over time for one or more runs."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Distance from the tool tip to the target",
                                        "True clearance: spacecraft and arm (solid) · obstacle (thin)"))
    for k, tr in enumerate(traces):
        col = SERIES[list(CONTROLLERS).index(tr["controller"])]
        fig.add_scatter(x=tr["t"], y=100 * tr["d_tcp"], mode="lines", name=tr["controller"], legendgroup=tr["controller"],
                        line={"color": col, "width": 2}, row=1, col=1)
        fig.add_scatter(x=tr["t"], y=100 * np.minimum(tr["d_body"], tr["d_self"]), mode="lines",
                        name=f"{tr['controller']}: spacecraft/arm", legendgroup=tr["controller"], showlegend=False,
                        line={"color": col, "width": 2}, row=2, col=1)
        if np.any(tr["d_obs"] < 0.299):
            fig.add_scatter(x=tr["t"], y=100 * np.where(tr["d_obs"] < 0.299, tr["d_obs"], np.nan), mode="lines",
                            name=f"{tr['controller']}: obstacle", legendgroup=tr["controller"], showlegend=False,
                            line={"color": col, "width": 1}, row=2, col=1)
        if len(traces) == 1 and tr["interventions"]:
            on = tr["t"][tr["shield"] > 0]
            fig.add_scatter(x=on, y=np.full(len(on), -1.5), mode="markers", name="shield intervened",
                            marker={"color": CRITICAL, "size": 6, "symbol": "line-ns-open"}, row=2, col=1)
    fig.add_hline(y=2, line={"color": MARGIN_INK, "width": 1}, row=2, col=1, annotation_text="2 cm near-miss line",
                  annotation_position="top right")
    fig.add_hline(y=0, line={"color": CRITICAL, "width": 1}, row=2, col=1)
    fig.add_hline(y=2, line={"color": MARGIN_INK, "width": 1}, row=1, col=1, annotation_text="2 cm success radius",
                  annotation_position="top right")
    s = traces[0]["sampled"]
    seen = [tr["sampled"]["obstacle_seen_s"] for tr in traces if tr["sampled"]["obstacle_seen_s"] is not None]
    for t_ev, label in ((min(seen) if seen else None, "obstacle appears"), (s["slip_s"], "encoder slip")):
        if t_ev is not None:
            for row in (1, 2):
                fig.add_vline(x=t_ev, line={"color": MARGIN_INK, "width": 1}, row=row, col=1)
            fig.add_annotation(x=t_ev, y=0.0, xref="x2", yref="paper", text=label, showarrow=False, xanchor="left",
                               yanchor="bottom", xshift=3, font={"size": 11, "color": MARGIN_INK})
    _chart_layout(fig, None, None, None, height=560)
    fig.update_yaxes(title_text="cm", row=1, col=1)
    fig.update_yaxes(title_text="cm", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(margin={"t": 40, "b": 90}, legend={"y": -0.12})
    return fig


def mission_scene(tr: dict, k: int, height: int = 560) -> go.Figure:
    """World-frame 3D view of step k: the spacecraft where it has drifted/rotated to, the arm, target, obstacle."""
    m = models()
    k = int(np.clip(k, 0, len(tr["t"]) - 1))
    R, pos = _quat_to_R(tr["base_q"][k]), tr["base_p"][k]
    traces = [scene.spacecraft_trace(m.cfg, R, pos), scene.arm_trace(m.kin, tr["q"][k], R, pos),
              scene.line_trace(tr["tcp"][: k + 1], SERIES[list(CONTROLLERS).index(tr["controller"])], "tool-tip trail",
                               width=5),
              scene.point_trace(tr["target"], scene.COLORS["target"], "target (fixed in space)", size=8)]
    if tr["obs_center"][k] is not None:
        traces.append(scene.sphere_trace(tr["obs_center"][k], tr["obs_radius"][k], scene.COLORS["obstacle"], "obstacle"))
    title = (f"{tr['controller']} · t = {tr['t'][k]:.1f} s · to target {100 * tr['d_tcp'][k]:.1f} cm · "
             f"spacecraft rotated {np.degrees(2 * np.arccos(min(1.0, abs(tr['base_q'][k][3])))):.1f}°")
    return go.Figure(traces, layout=scene.layout(title, height=height, uirevision="mission"))


def mission_summary(tr: dict) -> list:
    t_end = tr["t"][-1]
    return [tr["controller"], tr["outcome"], f"{t_end:.1f} s", f"{100 * tr['d_tcp'][-1]:.1f} cm",
            f"{100 * tr['min_clearance']:+.1f} cm", f"{tr['interventions']}" if "shield" in tr["controller"] else "—",
            f"{tr['base_rotation_deg']:.1f}°"]


SUMMARY_HEADERS = ["Controller", "Outcome", "Time", "Final dist.", "Min. clear.", "Shield acts", "Rotation"]
