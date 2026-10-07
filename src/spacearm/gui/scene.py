"""3D scene for the GUI (and figures): the spacecraft and the arm as triangle meshes, drawn with Plotly.

All geometry comes from the config and the exact FK (`ArmKinematics`), so what you see is exactly what the
planner, the DistanceNet and PyBullet reason about: boxes for the bus, panels and payload, a cylinder for the
pedestal and one capsule per arm link. Body frame unless a base pose (R, p) is given (Level 2, world frame).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch

from spacearm.robot_model import panel_centres

# Colours follow the PyBullet renders (reports/figures) so the GUI, the videos and the writeup look alike.
COLORS = {"bus": "#c9a227", "panel": "#26408b", "payload": "#b9b9c0", "pedestal": "#56565c",
          "arm_a": "#eb6834", "arm_b": "#ededed", "tcp": "#d03b3b", "target": "#0ca30c", "obstacle": "#e34948",
          "body_pair": "#d03b3b", "self_pair": "#4a3aa7", "ghost": "#9aa0a6"}
SURFACE = "#fcfcfb"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
# Fixed view volume (m, body frame) so the camera does not jump between updates; panels span y = +-2.15 m.
SCENE_RANGE = {"x": (-1.3, 1.9), "y": (-2.25, 2.25), "z": (-0.7, 2.3)}
_LIGHT = {"ambient": 0.55, "diffuse": 0.75, "specular": 0.15, "roughness": 0.6, "fresnel": 0.05}


# ------------------------------------------------------------------ primitive meshes (V (n, 3), F (m, 3))
def box_mesh(center, size):
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    V = c + h * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return V, F


def _frame(u):
    """Orthonormal (v, w) perpendicular to the unit vector u."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, tmp)
    v /= np.linalg.norm(v)
    return v, np.cross(u, v)


def _rings(profile, n_theta):
    """Surface of revolution around z from a list of (z, radius) rings; poles (radius 0) become single points."""
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    V, rows = [], []
    for z, r in profile:
        if r < 1e-12:
            rows.append([len(V)])
            V.append([0.0, 0.0, z])
        else:
            rows.append(list(range(len(V), len(V) + n_theta)))
            V.extend(np.stack([r * np.cos(th), r * np.sin(th), np.full(n_theta, z)], 1).tolist())
    F = []
    for a, b in zip(rows[:-1], rows[1:]):
        if len(a) == 1:
            F += [[a[0], b[k], b[(k + 1) % n_theta]] for k in range(n_theta)]
        elif len(b) == 1:
            F += [[a[k], b[0], a[(k + 1) % n_theta]] for k in range(n_theta)]
        else:
            for k in range(n_theta):
                k2 = (k + 1) % n_theta
                F += [[a[k], b[k], b[k2]], [a[k], b[k2], a[k2]]]
    return np.asarray(V, float), np.asarray(F, int)


def _place(V, a, u):
    v, w = _frame(u)
    return a + V[:, :1] * v + V[:, 1:2] * w + V[:, 2:3] * u


def capsule_mesh(a, b, r, n_theta=18, n_cap=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L if L > 1e-12 else np.array([0.0, 0.0, 1.0])
    phis = np.linspace(-np.pi / 2, 0.0, n_cap + 1)
    prof = [(r * np.sin(f), r * np.cos(f)) for f in phis] + [(L + r * np.sin(-f), r * np.cos(f)) for f in phis[::-1]]
    V, F = _rings(prof, n_theta)
    return _place(V, a, u), F


def cylinder_mesh(a, b, r, n_theta=24):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    V, F = _rings([(0.0, 0.0), (0.0, r), (L, r), (L, 0.0)], n_theta)
    return _place(V, a, (b - a) / L), F


def sphere_mesh(c, r, n_theta=20, n_phi=10):
    phis = np.linspace(-np.pi / 2, np.pi / 2, n_phi + 1)
    V, F = _rings([(r * np.sin(f), r * np.cos(f)) for f in phis], n_theta)
    return V + np.asarray(c, float), F


def _merge(parts):
    """[(V, F, color)] -> one mesh with per-face colours."""
    Vs, Fs, C, off = [], [], [], 0
    for V, F, col in parts:
        Vs.append(V)
        Fs.append(F + off)
        C += [col] * len(F)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), C


# ------------------------------------------------------------------ robot geometry
def spacecraft_mesh(cfg):
    """Bus, panels, payload(s) and pedestal in the body frame, as one mesh with face colours."""
    r = cfg["robot"]
    parts = [(*box_mesh([0, 0, 0], r["bus"]["size"]), COLORS["bus"])]
    for c in panel_centres(cfg).values():
        parts.append((*box_mesh(c, r["panels"]["size"]), COLORS["panel"]))
    for pl in r["payloads"]:
        parts.append((*box_mesh(pl["xyz"], pl["size"]), COLORS["payload"]))
    m = np.asarray(r["mount_xyz"], float)
    parts.append((*cylinder_mesh(m, m + [0, 0, r["pedestal"]["length"]], r["pedestal"]["radius"]), COLORS["pedestal"]))
    return _merge(parts)


def arm_geometry(kin, q):
    """Capsule segments a (7, 3), b (7, 3), radii (7,) and the TCP (3,) in the body frame (exact FK)."""
    with torch.no_grad():
        R, t = kin.forward(torch.as_tensor(np.asarray(q, float), dtype=kin.dtype))
        a, b, rad = kin.capsules_from(R, t)
    return a.double().numpy(), b.double().numpy(), rad.double().numpy(), t[-1].double().numpy()


def arm_mesh(kin, q, colors=None):
    a, b, rad, tcp = arm_geometry(kin, q)
    cols = colors or [COLORS["arm_a"] if i % 2 == 0 else COLORS["arm_b"] for i in range(len(a))]
    # Neighbouring capsules share a joint sphere; shrinking every other one by 1.5 % avoids z-fighting (display only).
    parts = [(*capsule_mesh(a[i], b[i], rad[i] * (1.0 if i % 2 == 0 else 0.985)), cols[i]) for i in range(len(a))]
    parts.append((*sphere_mesh(tcp, 0.022, 12, 6), COLORS["tcp"]))
    return _merge(parts)


def to_world(V, R=None, p=None):
    if R is None:
        return V
    return V @ np.asarray(R, float).T + np.asarray(p, float)


# ------------------------------------------------------------------ Plotly traces
def mesh_trace(V, F, colors, name, opacity=1.0, flat=False, showlegend=False):
    return go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], facecolor=colors,
                     opacity=opacity, flatshading=flat, lighting=_LIGHT, lightposition={"x": 1500, "y": -1200, "z": 3000},
                     name=name, hoverinfo="name", showlegend=showlegend, showscale=False)


def spacecraft_trace(cfg, R=None, p=None, opacity=1.0):
    V, F, C = spacecraft_mesh(cfg)
    return mesh_trace(to_world(V, R, p), F, C, "spacecraft", opacity=opacity, flat=True)


def arm_trace(kin, q, R=None, p=None, opacity=1.0, name="arm", ghost=False, color=None):
    cols = [color or COLORS["ghost"]] * kin.n_joints if (ghost or color) else None
    V, F, C = arm_mesh(kin, q, cols)
    return mesh_trace(to_world(V, R, p), F, C, name, opacity=opacity)


def sphere_trace(c, r, color, name, opacity=0.55):
    V, F = sphere_mesh(c, r, 24, 12)
    return mesh_trace(V, F, [color] * len(F), name, opacity=opacity)


def point_trace(p, color, name, size=7, symbol="diamond"):
    p = np.asarray(p, float).reshape(-1, 3)
    return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                        marker={"size": size, "color": color, "symbol": symbol, "line": {"color": SURFACE, "width": 2}},
                        hovertemplate=f"{name}<br>x %{{x:.2f}} m<br>y %{{y:.2f}} m<br>z %{{z:.2f}} m<extra></extra>")


def line_trace(P, color, name, width=5, showlegend=True, opacity=1.0):
    P = np.asarray(P, float)
    return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", name=name, opacity=opacity,
                        line={"color": color, "width": width}, showlegend=showlegend, hoverinfo="name")


def segment_trace(p0, p1, color, label):
    """Closest-point segment between two bodies, labelled with its length."""
    P = np.array([p0, p1], float)
    mid = P.mean(0)
    return go.Scatter3d(x=list(P[:, 0]) + [mid[0]], y=list(P[:, 1]) + [mid[1]], z=list(P[:, 2]) + [mid[2]],
                        mode="lines+markers+text", text=["", "", label], textposition="top center",
                        textfont={"color": INK, "size": 13},
                        marker={"size": [4, 4, 0], "color": color}, line={"color": color, "width": 6},
                        name=label, hoverinfo="name", showlegend=False)


def layout(title=None, camera=None, height=560, uirevision="scene", show_axes=True, legend=True):
    ax = {"showbackground": False, "gridcolor": GRID, "zerolinecolor": GRID, "color": MUTED,
          "title": {"font": {"size": 11, "color": MUTED}}, "tickfont": {"size": 9, "color": MUTED},
          "showspikes": False, "visible": show_axes}
    span = {k: v[1] - v[0] for k, v in SCENE_RANGE.items()}
    s = max(span.values())
    return go.Layout(
        title={"text": title, "x": 0.02, "y": 0.97, "font": {"size": 15, "color": INK}} if title else None,
        height=height, margin={"l": 0, "r": 0, "t": 36 if title else 6, "b": 0}, paper_bgcolor=SURFACE,
        uirevision=uirevision, showlegend=legend,
        legend={"x": 0.01, "y": 0.02, "bgcolor": "rgba(252,252,251,0.75)", "font": {"size": 11, "color": INK}},
        scene={"xaxis": {**ax, "range": list(SCENE_RANGE["x"]), "title": {**ax["title"], "text": "x (m)"}},
               "yaxis": {**ax, "range": list(SCENE_RANGE["y"]), "title": {**ax["title"], "text": "y (m)"}},
               "zaxis": {**ax, "range": list(SCENE_RANGE["z"]), "title": {**ax["title"], "text": "z (m)"}},
               "aspectmode": "manual", "aspectratio": {k: span[k] / s * 1.6 for k in span},
               "camera": camera or {"eye": {"x": 0.85, "y": -0.75, "z": 0.45}, "center": {"x": 0.12, "y": 0.02, "z": 0.06}},
               "bgcolor": SURFACE})


def robot_figure(cfg, kin, q, extra=(), title=None, R=None, p=None, height=560, uirevision="scene"):
    """Spacecraft + arm at joint angles q (+ any extra traces)."""
    fig = go.Figure([spacecraft_trace(cfg, R, p), arm_trace(kin, q, R, p), *extra],
                    layout=layout(title, height=height, uirevision=uirevision))
    return fig
