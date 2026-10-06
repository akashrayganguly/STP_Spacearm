"""Interactive PyBullet GUI: one slider per joint, live clearance read-out (workstation only, needs a display).

    python scripts/view_robot.py [--config configs/default.yaml]

Fold the arm onto the deck (joint2 and joint4 towards -90 deg) and watch d_body turn negative.
On a headless machine use scripts/render_robot.py instead.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

from spacearm.config import load_config
from spacearm.robot_model import ARM_JOINTS
from spacearm.sim import SpaceRobotSim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rate", type=float, default=10.0, help="clearance updates per second")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim = SpaceRobotSim(cfg, gui=True)
    cid = sim.cid
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=cid)
    p.resetDebugVisualizerCamera(3.0, 40.0, -22.0, [0.0, 0.0, 0.55], physicsClientId=cid)
    sliders = [p.addUserDebugParameter(name, float(lo), float(hi), 0.0, physicsClientId=cid)
               for name, lo, hi in zip(ARM_JOINTS, sim.lower, sim.upper)]
    text_id, last = -1, None
    print("Move the sliders; close the window or press Ctrl+C to quit.")
    try:
        while p.isConnected(physicsClientId=cid):
            q = np.array([p.readUserDebugParameter(s, physicsClientId=cid) for s in sliders])
            if last is None or np.abs(q - last).max() > 1e-6:
                sim.set_q(q)
                d_body, d_self = sim.min_distances()
                msg = f"d_body {d_body * 100:+.1f} cm   d_self {d_self * 100:+.1f} cm"
                msg += "   COLLISION" if min(d_body, d_self) <= 0 else ""
                text_id = p.addUserDebugText(msg, [0.0, 0.0, 2.3], textColorRGB=[1, 0, 0] if "COLL" in msg else [0, 0, 0],
                                             textSize=1.4, replaceItemUniqueId=text_id, physicsClientId=cid)
                print(f"q = {np.array2string(np.degrees(q), precision=0, suppress_small=True)} deg   {msg}")
                last = q
            time.sleep(1.0 / args.rate)
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
