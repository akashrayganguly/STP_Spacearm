"""Write the robot URDF generated from the config (for viewing; the simulator builds its own copy).

    python scripts/make_urdf.py [--config configs/default.yaml] [--out assets/space_robot.urdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from spacearm.config import ROOT, load_config
from spacearm.robot_model import ARM_JOINTS, spacecraft_links, total_mass, write_urdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--out", default=None, help="output path (default: paths.urdf from the config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else ROOT / cfg["paths"]["urdf"]
    write_urdf(cfg, out)
    print(f"wrote {out}")
    print(f"  spacecraft parts: {', '.join(spacecraft_links(cfg))}")
    print(f"  arm joints:       {', '.join(ARM_JOINTS)} (axes {''.join(cfg['robot']['arm']['axes'])})")
    print(f"  total mass:       {total_mass(cfg):.2f} kg")


if __name__ == "__main__":
    main()
