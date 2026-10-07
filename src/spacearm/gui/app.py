"""spacearm interactive explorer (Gradio web app).

    python scripts/gui.py            ->  opens http://127.0.0.1:7860 in your browser

Tab 1  Pose & clearance (Level 1): set the 7 joint angles; see the arm, d_body / d_self from PyBullet (truth)
       and from the DistanceNet (prediction), and the closest link pairs.
Tab 2  Path planner (Level 1): start pose + target point -> every planner stage, verified in PyBullet.
Tab 3  Mission simulator (Level 2): start, target, controller and surprises (obstacle, faults, sensor noise)
       -> a closed-loop episode on the free-floating spacecraft, as a video with charts and a 3D replay.
"""
from __future__ import annotations

import tempfile

import gradio as gr
import numpy as np

from spacearm.gui import backend as B

AXES = ["z", "y", "z", "y", "z", "y", "z"]
# Default planner query: hard-set query 42 of reports/level1/stages.json (the baseline and TrajNet alone collide;
# refine clears it, IK polish lands the tip).
HARD_START = [-76.4, -93.1, 82.7, -69.7, 46.4, -45.3, -81.6]
HARD_TARGET = [-0.133, -0.069, 0.907]
TRAINING = {"radius": 12.0, "drift": 3.0, "bias": 2.0, "motor": 0.7, "noise": 1.0}
PRESETS = {
    # obstacle, radius cm, appear s, drift cm/s, bias deg, motor min, slip on, slip s, noise x
    "S1 nominal: no surprises": ("none", 8.0, 2.5, 0.0, 0.0, 1.0, False, 6.0, 0.0),
    "S2 obstacle: drifting ball": ("drifting", 8.0, 2.5, 2.0, 0.0, 1.0, False, 6.0, 0.0),
    "S3 hardware: faults + noise": ("none", 8.0, 2.5, 0.0, 2.0, 0.7, True, 6.0, 1.0),
    "S4 everything at full difficulty": ("drifting", 8.0, 2.5, 2.0, 2.0, 0.7, True, 6.0, 1.0),
    "Stress: big fast ball, bad sensors": ("drifting", 16.0, 2.0, 5.0, 4.0, 0.7, True, 4.0, 3.0),
}
CSS = """
.gradio-container {max-width: 1480px !important}
#hero {padding: 18px 22px; border-radius: 14px; background: linear-gradient(120deg, #0d366b 0%, #184f95 60%, #2a78d6 100%);
       color: #fff; margin-bottom: 6px}
#hero h1 {color: #fff; font-size: 26px; margin: 0 0 4px 0; font-weight: 650}
#hero p {color: #dbe8fb; margin: 0; font-size: 14px}
#hero .chips {margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap}
#hero .chip {background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
             padding: 3px 11px; font-size: 12.5px; color: #fff}
.tiles {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 2px 0 6px 0}
.tile {background: #ffffff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 10px 14px}
.tile .label {color: #52514e; font-size: 12px; text-transform: uppercase; letter-spacing: .04em}
.tile .value {color: #0b0b0b; font-size: 22px; font-weight: 650; margin-top: 2px}
.tile .sub {color: #898781; font-size: 12.5px; margin-top: 2px}
.status-safe .value {color: #006300} .status-close .value {color: #a36400} .status-bad .value {color: #b42323}
.note {color: #52514e; font-size: 13px}
.stress {color: #a36400; font-size: 13px}
"""


def _tile(label, value, sub="", cls=""):
    return f'<div class="tile {cls}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def _cm(d, clip=0.30):
    return f"≥ {100 * clip:.0f} cm" if d >= clip - 1e-9 else f"{100 * d:+.1f} cm"


# ------------------------------------------------------------------ tab 1
def pose_view(*q):
    rep = B.pose_report(q)
    st = rep["status"]
    cls = "status-bad" if st.startswith("COLLISION") else ("status-close" if st.startswith("CLOSE") else "status-safe")
    icon = "⛔" if cls == "status-bad" else ("⚠️" if cls == "status-close" else "✅")
    body = f"closest: {rep['body']['links'][0]} ↔ {rep['body']['links'][1]}" if rep["body"] else "nothing within 30 cm"
    selfp = f"closest: {rep['self']['links'][0]} ↔ {rep['self']['links'][1]}" if rep["self"] else "nothing within 30 cm"
    tiles = ('<div class="tiles">'
             + _tile("Status", f"{icon} {st.split(' (')[0]}", "true min. clearance " + _cm(min(rep['d_body'], rep['d_self'])), cls)
             + _tile("d_body · arm ↔ spacecraft", _cm(rep["d_body"]), f"DistanceNet {100 * rep['pred_body']:+.1f} cm · {body}")
             + _tile("d_self · arm ↔ arm", _cm(rep["d_self"]), f"DistanceNet {100 * rep['pred_self']:+.1f} cm · {selfp}")
             + _tile("Tool tip (body frame)", f"{rep['tcp'][0]:+.2f}, {rep['tcp'][1]:+.2f}, {rep['tcp'][2]:+.2f}",
                     "x, y, z in metres") + "</div>")
    return B.pose_figure(rep), tiles


def preset_pose(kind, counter):
    counter = int(counter) + 1
    q = [0.0] * 7 if kind == "zero" else B.random_pose(kind, 1000 + counter)
    return [*q, counter]


# ------------------------------------------------------------------ tab 2
def plan_view(*args):
    q, (x, y, z), baseline = args[:7], args[7:10], args[10]
    rep0 = B.pose_report(q)
    if min(rep0["d_body"], rep0["d_self"]) <= 0:
        raise gr.Error("The start pose collides. Pick another start (e.g. 'Random safe start').")
    ok, msg = B.check_target([x, y, z])
    if not ok:
        raise gr.Error(msg)
    rep = B.plan_report(q, [x, y, z], baseline=baseline)
    final = rep["rows"][-1]
    head = (f"**{'✅ Planned and verified' if final['success'] else '❌ No verified path'}** · {msg} · "
            f"start clearance {_cm(min(rep0['d_body'], rep0['d_self']))}")
    return (head, B.plan_figure(rep), B.plan_table(rep), B.plan_clearance_chart(rep), B.plan_joint_chart(rep))


def random_target(*q):
    rng = np.random.default_rng(int(abs(sum(q)) * 1000) % 2**31)
    for _ in range(200):
        tq = B.random_pose("safe", int(rng.integers(0, 2**31 - 1)))
        t = B.tcp_of(np.radians(tq))
        ok, _ = B.check_target(t)
        if ok and np.linalg.norm(t - B.tcp_of(np.radians(q))) > 0.2 and np.linalg.norm(t - B.shoulder_and_reach()[0]) <= 0.85 * B.shoulder_and_reach()[1]:
            return [round(float(v), 3) for v in t]
    raise gr.Error("Could not find a reachable target; try again.")


# ------------------------------------------------------------------ tab 3
def apply_preset(name):
    o, r, ts, v, b, mm, slip, sl, n = PRESETS[name]
    return o, r, ts, v, b, mm, slip, sl, n


def stress_note(r, v, b, mm, n, obstacle):
    out = []
    if obstacle != "none" and r > TRAINING["radius"]:
        out.append(f"ball radius {r:.0f} cm (trained ≤ 12)")
    if obstacle == "drifting" and v > TRAINING["drift"]:
        out.append(f"drift {v:.1f} cm/s (trained ≤ 3)")
    if b > TRAINING["bias"]:
        out.append(f"encoder bias {b:.1f}° (trained ≤ 2)")
    if mm < TRAINING["motor"]:
        out.append(f"motor strength {100 * mm:.0f} % (trained ≥ 70)")
    if n > TRAINING["noise"]:
        out.append(f"noise ×{n:.1f} (trained ≤ ×1)")
    return ("⚠️ Beyond the training range: " + ", ".join(out) + ". Expect lower success.") if out else \
        "All settings are inside the training range."


def mission_view(controller, seed, start_src, target_src, source, obstacle, radius, appear, drift, bias, motor,
                 slip_on, slip_t, noise, *planner_inputs):
    q_plan, t_plan = planner_inputs[:7], planner_inputs[7:10]
    start = list(q_plan) if start_src == "Planner tab start pose" else None
    target = list(t_plan) if target_src == "Planner tab target" else None
    if start is not None:
        rep = B.pose_report(start)
        if min(rep["d_body"], rep["d_self"]) <= 0.02:
            raise gr.Error("The planner-tab start pose is within 2 cm of a collision; pick a safer start.")
    if target is not None:
        ok, msg = B.check_target(target)
        if not ok:
            raise gr.Error(msg)
    spec = B.MissionSpec(seed=int(seed), start_deg=start, target=target,
                         source="manual" if source == "Set by hand (below)" else source.split(" · ")[0],
                         obstacle=obstacle, radius_cm=radius, appear_s=appear, drift_cm_s=drift, bias_deg=bias,
                         motor_min=motor, slip_s=slip_t if slip_on else None, noise=noise)
    names = list(B.CONTROLLERS) if controller == "Compare all four" else [controller]
    traces = [B.run_mission(c, spec) for c in names]
    video = B.grid_video(traces) if len(traces) > 1 else B.save_video(traces[0]["frames"])
    for tr in traces:
        tr["frames"] = None                                   # keep the session state small
    s = traces[0]["sampled"]
    what = (f"Start {np.round(s['start_deg'], 0).astype(int).tolist()}° · target "
            f"({s['target'][0]:+.2f}, {s['target'][1]:+.2f}, {s['target'][2]:+.2f}) m · "
            + obstacle_text(s, traces)
            + f"max |encoder bias| {max(abs(v) for v in s['bias_deg']):.1f}° · weakest motor {100 * min(s['gains']):.0f} % · "
            + (f"slip at {s['slip_s']:.1f} s · " if s["slip_s"] else "no slip · ")
            + f"encoder noise {s['noise_enc_deg']:.2f}°")
    last = len(traces[-1]["t"]) - 1
    return (video, [B.mission_summary(tr) for tr in traces], B.mission_chart(traces), what, traces,
            gr.update(maximum=last, value=last), gr.update(choices=names, value=names[-1]),
            B.mission_scene(traces[-1], last))


def obstacle_text(s, traces):
    if not s["obstacle_s"]:
        return "no obstacle · "
    if s["obstacle_s"] > max(tr["t"][-1] for tr in traces):
        return f"obstacle due at {s['obstacle_s']:.1f} s (every run ended before) · "
    seen = [tr["sampled"]["obstacle_seen_s"] for tr in traces]
    if all(v is None for v in seen):
        return (f"obstacle due at {s['obstacle_s']:.1f} s but no free spot between the hand and the target "
                "(training rule), so none appeared · ")
    return f"obstacle appeared at {s['obstacle_s']:.1f} s · "


def scrub(traces, which, k):
    if not traces:
        return None
    tr = next((t for t in traces if t["controller"] == which), traces[-1])
    return B.mission_scene(tr, int(min(k, len(tr["t"]) - 1)))


# ------------------------------------------------------------------ layout
def build() -> gr.Blocks:
    lo, hi = B.joint_limits_deg()
    with gr.Blocks(title="spacearm explorer") as demo:
        gr.HTML('<div id="hero"><h1>spacearm · interactive explorer</h1>'
                '<p>A 7-joint arm on a free-floating spacecraft: a neural planner (Level 1) and a safe '
                'reinforcement-learning controller (Level 2). Every number here is computed live by the trained models.</p>'
                '<div class="chips"><span class="chip">DistanceNet: 0.68 cm mean error</span>'
                '<span class="chip">Planner: 100 % verified, ~0.1 s</span>'
                '<span class="chip">Level 2, everything on: 58 % success, 0 % collisions</span></div></div>')
        with gr.Tabs():
            # ---------------------------------------------------------- tab 1
            with gr.Tab("① Pose & clearance"):
                gr.Markdown("Move the joints. **d_body** is the shortest gap between the arm and the spacecraft, "
                            "**d_self** between two arm links (negative = overlapping). The big numbers are the "
                            "PyBullet truth; *DistanceNet* is what the network predicts from the 7 angles alone. "
                            "The coloured segments join the closest points.", elem_classes="note")
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        q1 = [gr.Slider(lo[i], hi[i], value=[0, 45, 0, 60, 0, 45, 0][i], step=0.1,
                                        label=f"Joint {i + 1} · axis {AXES[i]} (deg)") for i in range(7)]
                        with gr.Row():
                            b_zero = gr.Button("Zero pose", size="sm")
                            b_safe = gr.Button("Random safe", size="sm")
                        with gr.Row():
                            b_close = gr.Button("Random close call", size="sm")
                            b_coll = gr.Button("Random collision", size="sm")
                        counter = gr.Number(value=0, visible=False)
                    with gr.Column(scale=3):
                        tiles = gr.HTML()
                        plot1 = gr.Plot(label="Arm and spacecraft (drag to rotate, scroll to zoom)")
                for s in q1:
                    s.release(pose_view, q1, [plot1, tiles])
                for btn, kind in ((b_zero, "zero"), (b_safe, "safe"), (b_close, "close"), (b_coll, "colliding")):
                    btn.click(lambda c, k=kind: preset_pose(k, c), counter, [*q1, counter]).then(pose_view, q1, [plot1, tiles])

            # ---------------------------------------------------------- tab 2
            with gr.Tab("② Path planner"):
                gr.Markdown("Choose a start pose and a target point (body frame, metres). The planner runs stage by "
                            "stage: **TrajNet** proposes a smooth path in one pass, **refine** nudges its 35 numbers "
                            "away from near-collisions (DistanceNet), **IK polish** puts the tool tip on the target, "
                            "and the **full planner** checks the result in PyBullet (retrying once if needed).",
                            elem_classes="note")
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        q2 = [gr.Slider(lo[i], hi[i], value=HARD_START[i], step=0.1,
                                        label=f"Start · joint {i + 1} (deg)") for i in range(7)]
                        with gr.Row():
                            b_from1 = gr.Button("Use pose from tab ①", size="sm")
                            b_rstart = gr.Button("Random safe start", size="sm")
                        with gr.Row():
                            tx = gr.Number(HARD_TARGET[0], label="target x (m)", step=0.05)
                            ty = gr.Number(HARD_TARGET[1], label="target y (m)", step=0.05)
                            tz = gr.Number(HARD_TARGET[2], label="target z (m)", step=0.05)
                        with gr.Row():
                            b_rtarget = gr.Button("Random reachable target", size="sm")
                            b_tip1 = gr.Button("Tool tip of tab ① pose", size="sm")
                        base_cb = gr.Checkbox(True, label="Also show the classical baseline (IK + straight line)")
                        b_plan = gr.Button("Plan", variant="primary")
                        c2 = gr.Number(value=0, visible=False)
                    with gr.Column(scale=3):
                        head2 = gr.Markdown()
                        plot2 = gr.Plot(label="Tool-tip path of every stage · grey ghosts: start pose and the arm 1/3 and 2/3 along the final path · solid: goal pose")
                        table2 = gr.Dataframe(headers=["Stage", "PyBullet verdict", "Min. clearance (true)",
                                                       "Reach error", "Time"], interactive=False)
                        with gr.Row():
                            chart2a = gr.Plot(show_label=False)
                            chart2b = gr.Plot(show_label=False)
                plan_inputs = [*q2, tx, ty, tz, base_cb]
                b_plan.click(plan_view, plan_inputs, [head2, plot2, table2, chart2a, chart2b])
                b_from1.click(lambda *q: list(q), q1, q2)
                b_rstart.click(lambda c: preset_pose("safe", c), c2, [*q2, c2])
                b_rtarget.click(random_target, q2, [tx, ty, tz])
                b_tip1.click(lambda *q: [round(float(v), 3) for v in B.tcp_of(np.radians(q))], q1, [tx, ty, tz])

            # ---------------------------------------------------------- tab 3
            with gr.Tab("③ Mission simulator"):
                gr.Markdown("A full closed-loop episode at 10 Hz on the **free-floating** spacecraft: the target is fixed "
                            "in space, the spacecraft turns as the arm moves, and surprises happen. Pick a controller "
                            "and the surprises, then **Run**. *Compare all four* runs the identical episode with every "
                            "controller.", elem_classes="note")
                with gr.Row():
                    with gr.Column(scale=1, min_width=330):
                        ctrl = gr.Radio(["Reflex only", "Reflex + shield", "RL", "RL + shield", "Compare all four"],
                                        value="Compare all four", label="Controller")
                        with gr.Row():
                            seed = gr.Number(1, label="Seed (start, target, random details)", precision=0)
                            b_dice = gr.Button("🎲", size="sm", min_width=50)
                        with gr.Accordion("Start pose and target", open=False):
                            start_src = gr.Radio(["Random from the seed", "Planner tab start pose"],
                                                 value="Random from the seed", label="Start pose")
                            target_src = gr.Radio(["Random from the seed", "Planner tab target"],
                                                  value="Random from the seed", label="Target")
                        source = gr.Dropdown(["Set by hand (below)"] + [f"{k} · replay the evaluation episode"
                                                                        for k in B.SCENARIOS],
                                             value="Set by hand (below)", label="Surprises",
                                             info="Replay mode samples everything from the seed exactly as in the "
                                                  "evaluation: seeds 10000–10099 are the report's episodes.")
                        preset = gr.Dropdown(list(PRESETS), value="S4 everything at full difficulty",
                                             label="Preset (fills the settings below)")
                        with gr.Accordion("Obstacle", open=True):
                            obstacle = gr.Radio(["none", "static", "drifting"], value="drifting", label="Ball obstacle")
                            radius = gr.Slider(5, 20, 8, step=1, label="Radius (cm) · trained 5–12")
                            appear = gr.Slider(0.5, 15, 2.5, step=0.5, label="Appears at (s) · trained 1–5",
                                               info="Placed between the hand and the target at that moment.")
                            drift = gr.Slider(0, 6, 2, step=0.5, label="Drift speed (cm/s) · trained ≤ 3")
                        with gr.Accordion("Hardware faults", open=True):
                            bias = gr.Slider(0, 5, 2, step=0.25, label="Encoder bias, max per joint (deg) · trained ≤ 2")
                            motor = gr.Slider(0.5, 1.0, 0.7, step=0.05, label="Weakest motor strength · trained ≥ 0.7")
                            with gr.Row():
                                slip_on = gr.Checkbox(True, label="Encoder slip (±3° on one joint)")
                                slip_t = gr.Slider(0.5, 19, 6, step=0.5, label="Slip at (s)")
                        with gr.Accordion("Sensor noise", open=True):
                            noise = gr.Slider(0, 3, 1, step=0.25, label="Noise × training level",
                                              info="×1 = encoders 0.1°, camera 5 mm, gyro 0.002 rad/s")
                        stress = gr.Markdown(elem_classes="stress")
                        b_run = gr.Button("Run mission", variant="primary")
                    with gr.Column(scale=3):
                        gr.Markdown("**Episode video** · the camera follows the spacecraft · green dot = target "
                                    "(fixed in space) · red ball = obstacle", elem_classes="note")
                        video = gr.Video(show_label=False, autoplay=True, loop=True, height=560)
                        what = gr.Markdown(elem_classes="note")
                        table3 = gr.Dataframe(headers=B.SUMMARY_HEADERS, interactive=False)
                        chart3 = gr.Plot(show_label=False)
                        with gr.Row():
                            which = gr.Dropdown(["RL + shield"], value="RL + shield", label="3D replay of", scale=1)
                            k3 = gr.Slider(0, 200, 0, step=1, label="Time step (×0.1 s)", scale=3)
                        plot3 = gr.Plot(show_label=False)
                traces = gr.State([])
                surprise_inputs = [obstacle, radius, appear, drift, bias, motor, slip_on, slip_t, noise]
                preset.change(apply_preset, preset, surprise_inputs)
                for comp in (radius, drift, bias, motor, noise, obstacle):
                    comp.change(stress_note, [radius, drift, bias, motor, noise, obstacle], stress)
                b_dice.click(lambda: int(np.random.default_rng().integers(0, 99999)), None, seed)
                b_run.click(mission_view, [ctrl, seed, start_src, target_src, source, *surprise_inputs, *q2, tx, ty, tz],
                            [video, table3, chart3, what, traces, k3, which, plot3])
                k3.release(scrub, [traces, which, k3], plot3)
                which.change(scrub, [traces, which, k3], plot3)

            with gr.Tab("About"):
                gr.Markdown(ABOUT)
        demo.load(pose_view, q1, [plot1, tiles])
        demo.load(stress_note, [radius, drift, bias, motor, noise, obstacle], stress)
    return demo


ABOUT = """
### What you are looking at
* **Level 1 (tabs ① and ②)** plans in the spacecraft's own frame, where collisions depend only on the joint angles.
  *DistanceNet* (136k parameters) predicts the clearances from the 7 angles; *TrajNet* (552k parameters) proposes a
  smooth path (a degree-7 Bezier curve with 8 control points); a short optimisation (*refine*) and an inverse-kinematics
  step (*IK polish*) finish it, and PyBullet checks it.
* **Level 2 (tab ③)** flies the arm closed-loop at 10 Hz on the free-floating spacecraft. The *reflex* is a classical
  controller (reach + stay away from the spacecraft + push away from obstacles); the *RL* policy (8k parameters,
  PPO-Lagrangian) adds a learned correction on top; the *shield* checks every command against the models and slows or
  redirects it when a margin would be broken.
* Settings beyond the training range are allowed on purpose (stress tests) and flagged.

Full story, figures and every evaluation table: **docs/WRITEUP.md**. Numbers: **reports/**.
"""


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines on your network")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    args = ap.parse_args(argv)
    B.models()                                                     # load the networks before the first click
    B.plan_report(HARD_START, HARD_TARGET, baseline=False)          # warm-up: the first autograd call is slow
    demo = build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser, share=args.share,
                show_error=True, allowed_paths=[tempfile.gettempdir()],
                theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone",
                                     font=[gr.themes.Font(f) for f in ("ui-sans-serif", "system-ui", "Segoe UI",
                                                                       "sans-serif")]),
                css=CSS, footer_links=[])


if __name__ == "__main__":
    main()
