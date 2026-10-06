"""Robot description: free-floating spacecraft (bus + panels + payloads) carrying one 7-DOF arm.

The URDF is generated from the config (single source of truth), so geometry, masses and
joint limits can never drift away from the numbers that the kinematics and tests use.

Frames: body frame B = bus centre (= bus centre of mass), z up through the arm-mount face,
y along the solar panels. Each arm link frame sits on its joint; the link's capsule runs
from z = 0 to z = length along the link's own +z axis (same segment as `ArmKinematics.capsules`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_LINKS = [f"link{i}" for i in range(1, 8)]
TCP_LINK = "tcp"

_AXIS_XYZ = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}
# Cosmetic only (visual shapes for the offscreen renderer / GUI).
_COLORS = {"bus": "0.85 0.70 0.25 1", "panel": "0.15 0.25 0.65 1", "payload": "0.75 0.75 0.78 1",
           "pedestal": "0.35 0.35 0.38 1", "arm_even": "0.92 0.48 0.12 1", "arm_odd": "0.95 0.95 0.95 1",
           "tcp": "0.9 0.1 0.1 1"}
_TCP_VISUAL_RADIUS = 0.02


# ------------------------------------------------------------------ config helpers
def joint_limits(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) joint limits in radians, shape (7,) each."""
    arm = cfg["robot"]["arm"]
    return np.radians(np.asarray(arm["lower_deg"], float)), np.radians(np.asarray(arm["upper_deg"], float))


def spacecraft_links(cfg: dict) -> list[str]:
    """Names of the rigid spacecraft parts the arm must never touch (the bus is the base link)."""
    return ["bus", "panel_left", "panel_right"] + [pl["name"] for pl in cfg["robot"]["payloads"]] + ["pedestal"]


def total_mass(cfg: dict) -> float:
    """Total mass of the spacecraft + arm (kg), as built into the URDF."""
    r = cfg["robot"]
    return float(r["bus"]["mass"] + 2 * r["panels"]["mass"] + sum(pl["mass"] for pl in r["payloads"])
                 + r["pedestal"]["mass"] + sum(r["arm"]["link_masses"]) + r["arm"]["tcp_mass"])


def panel_centres(cfg: dict) -> dict[str, list[float]]:
    """Centres of the two solar panels in the body frame (left = +y, right = -y)."""
    r = cfg["robot"]
    y = r["bus"]["size"][1] / 2 + r["panels"]["gap"] + r["panels"]["size"][1] / 2
    return {"panel_left": [0.0, y, 0.0], "panel_right": [0.0, -y, 0.0]}


# ------------------------------------------------------------------ inertia formulas
def _box_inertia(m: float, size) -> tuple[float, float, float]:
    x, y, z = size
    return m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12


def _cylinder_inertia(m: float, r: float, length: float) -> tuple[float, float, float]:
    """Solid cylinder along z (used for the pedestal and as the capsule-link approximation)."""
    ixx = m * (3 * r * r + length * length) / 12
    return ixx, ixx, m * r * r / 2


# ------------------------------------------------------------------ URDF text
def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _inertial(m: float, inertia, xyz=(0.0, 0.0, 0.0)) -> str:
    ixx, iyy, izz = inertia
    return (f'    <inertial>\n      <origin xyz="{_fmt(xyz)}" rpy="0 0 0"/>\n      <mass value="{m:.6g}"/>\n'
            f'      <inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>\n'
            f"    </inertial>\n")


def _shape(kind: str, geometry: str, color: str, xyz=(0.0, 0.0, 0.0)) -> str:
    vis = (f'    <visual>\n      <origin xyz="{_fmt(xyz)}" rpy="0 0 0"/>\n      <geometry>{geometry}</geometry>\n'
           f'      <material name="{kind}"><color rgba="{color}"/></material>\n    </visual>\n')
    col = (f'    <collision>\n      <origin xyz="{_fmt(xyz)}" rpy="0 0 0"/>\n      <geometry>{geometry}</geometry>\n'
           f"    </collision>\n")
    return vis + col


def _box_link(name: str, size, mass: float, color: str) -> str:
    return (f'  <link name="{name}">\n' + _inertial(mass, _box_inertia(mass, size))
            + _shape(name, f'<box size="{_fmt(size)}"/>', color) + "  </link>\n")


def _fixed_joint(name: str, parent: str, child: str, xyz) -> str:
    return (f'  <joint name="{name}" type="fixed">\n    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
            f'    <origin xyz="{_fmt(xyz)}" rpy="0 0 0"/>\n  </joint>\n')


def build_urdf(cfg: dict) -> str:
    """URDF text of the spacecraft + arm described by `cfg["robot"]`."""
    r = cfg["robot"]
    arm = r["arm"]
    lo, hi = joint_limits(cfg)
    parts = ['<?xml version="1.0"?>\n<robot name="space_robot">\n']

    # Spacecraft: the bus is the (free-floating) base link; everything else is fixed to it.
    parts.append(_box_link("bus", r["bus"]["size"], r["bus"]["mass"], _COLORS["bus"]))
    for name, centre in panel_centres(cfg).items():
        parts.append(_box_link(name, r["panels"]["size"], r["panels"]["mass"], _COLORS["panel"]))
        parts.append(_fixed_joint(f"{name}_joint", "bus", name, centre))
    for pl in r["payloads"]:
        parts.append(_box_link(pl["name"], pl["size"], pl["mass"], _COLORS["payload"]))
        parts.append(_fixed_joint(f"{pl['name']}_joint", "bus", pl["name"], pl["xyz"]))

    ped = r["pedestal"]
    half = (0.0, 0.0, ped["length"] / 2)
    parts.append('  <link name="pedestal">\n'
                 + _inertial(ped["mass"], _cylinder_inertia(ped["mass"], ped["radius"], ped["length"]), half)
                 + _shape("pedestal", f'<cylinder radius="{ped["radius"]:.6g}" length="{ped["length"]:.6g}"/>',
                          _COLORS["pedestal"], half)
                 + "  </link>\n")
    parts.append(_fixed_joint("pedestal_joint", "bus", "pedestal", r["mount_xyz"]))

    # Arm: joint i sits at the end of the previous link (pedestal for joint 1), capsule along +z.
    parent, offset = "pedestal", ped["length"]
    for i in range(7):
        L, rad, m = arm["link_lengths"][i], arm["link_radii"][i], arm["link_masses"][i]
        half = (0.0, 0.0, L / 2)
        color = _COLORS["arm_even"] if i % 2 == 0 else _COLORS["arm_odd"]
        parts.append(f'  <link name="{ARM_LINKS[i]}">\n'
                     + _inertial(m, _cylinder_inertia(m, rad, L), half)
                     + _shape(ARM_LINKS[i], f'<capsule radius="{rad:.6g}" length="{L:.6g}"/>', color, half)
                     + "  </link>\n")
        parts.append(f'  <joint name="{ARM_JOINTS[i]}" type="revolute">\n    <parent link="{parent}"/>\n'
                     f'    <child link="{ARM_LINKS[i]}"/>\n    <origin xyz="0 0 {offset:.6g}" rpy="0 0 0"/>\n'
                     f'    <axis xyz="{_AXIS_XYZ[arm["axes"][i]]}"/>\n'
                     f'    <limit lower="{lo[i]:.10g}" upper="{hi[i]:.10g}" effort="{arm["max_torque"]:.6g}" '
                     f'velocity="{arm["max_velocity"]:.6g}"/>\n'
                     f'    <dynamics damping="0" friction="0"/>\n  </joint>\n')
        parent, offset = ARM_LINKS[i], L

    # Tool centre point: a (nearly) massless frame with a visual marker but no collision shape.
    m_tcp = arm["tcp_mass"]
    i_tcp = 0.4 * m_tcp * _TCP_VISUAL_RADIUS ** 2
    parts.append(f'  <link name="{TCP_LINK}">\n' + _inertial(m_tcp, (i_tcp, i_tcp, i_tcp))
                 + f'    <visual>\n      <geometry><sphere radius="{_TCP_VISUAL_RADIUS}"/></geometry>\n'
                   f'      <material name="tcp"><color rgba="{_COLORS["tcp"]}"/></material>\n    </visual>\n'
                 + "  </link>\n")
    parts.append(_fixed_joint("tcp_joint", ARM_LINKS[-1], TCP_LINK, (0.0, 0.0, arm["tool_offset"])))
    parts.append("</robot>\n")
    return "".join(parts)


def write_urdf(cfg: dict, path: str | Path) -> Path:
    """Write the URDF to `path` (parent folders are created) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_urdf(cfg), encoding="utf-8")
    return path
