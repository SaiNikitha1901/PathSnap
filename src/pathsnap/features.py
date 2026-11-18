"""Feature engineering for (GPS point, candidate arc) pairs.

None of these features look at the reference route, so they can be computed for
any new trip exactly as at training time (no label leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff, bearing_deg, point_segment
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "cos_heading", "heading_diff", "heading_valid",
    "proj_t", "arc_len", "from_out_deg", "to_out_deg",
    "speed", "dt_prev", "win_dist", "win_dist_minus_min",
    "near_prev_best", "near_next_best",
]


def point_motion(pxy: np.ndarray, times: np.ndarray, w: int = 2):
    """Heading and speed of each point from its neighbours (i-w .. i+w).
    Heading is marked invalid when the vehicle barely moves (< 3 m)."""
    N = len(pxy)
    lo = np.clip(np.arange(N) - w, 0, N - 1)
    hi = np.clip(np.arange(N) + w, 0, N - 1)
    dx = pxy[hi, 0] - pxy[lo, 0]
    dy = pxy[hi, 1] - pxy[lo, 1]
    disp = np.hypot(dx, dy)
    dt = np.maximum(times[hi] - times[lo], 1e-6)
    heading = bearing_deg(dx, dy)
    valid = disp >= 3.0
    speed = np.where(hi > lo, disp / dt, 0.0)
    dt_prev = np.r_[0.0, np.diff(times)] if N else np.zeros(0)
    return heading, valid, speed, dt_prev


def build_features(cands: pd.DataFrame, pxy: np.ndarray, times: np.ndarray,
                   net: RoadNetwork, w: int = 2) -> pd.DataFrame:
    df = cands.copy()
    if len(df) == 0:
        for f in FEATURES:
            df[f] = []
        return df
    pt = df["pt"].to_numpy()
    arc = df["arc"].to_numpy()
    heading, valid, speed, dt_prev = point_motion(pxy, times, w)

    g = df.groupby("pt")["dist"]
    dmin = g.transform("min").to_numpy()
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = df["dist"] / (dmin + 1.0)
    df["n_cands"] = g.transform("size").to_numpy()

    hd = angle_diff(heading[pt], net.bearing[arc])
    df["heading_valid"] = valid[pt].astype(float)
    df["heading_diff"] = np.where(valid[pt], hd, 90.0)      # 90 = "no information"
    df["cos_heading"] = np.where(valid[pt], np.cos(np.radians(hd)), 0.0)
    df["proj_t"] = df["t"]
    df["arc_len"] = net.length[arc]
    df["from_out_deg"] = net.out_deg[net.arcs[arc, 0]]
    df["to_out_deg"] = net.out_deg[net.arcs[arc, 1]]
    df["speed"] = speed[pt]
    df["dt_prev"] = dt_prev[pt]

    # window distance: mean distance of the neighbouring points (i-w..i+w) to this arc.
    # A parallel road may be close to one noisy point but not to the whole window.
    N = len(pxy)
    acc = np.zeros(len(df))
    for o in range(-w, w + 1):
        j = np.clip(pt + o, 0, N - 1)
        d, _ = point_segment(pxy[j, 0], pxy[j, 1], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
        acc += d
    df["win_dist"] = acc / (2 * w + 1)
    df["win_dist_minus_min"] = df["win_dist"] - df.groupby("pt")["win_dist"].transform("min")

    # is this arc equal / adjacent to the nearest arc of the previous / next point?
    nearest = df.loc[df["dist_rank"] == 0, ["pt", "arc"]].set_index("pt")["arc"]
    best = np.full(N, -1, dtype=np.int64)
    best[nearest.index.to_numpy()] = nearest.to_numpy()
    df["near_prev_best"] = _adjacent(net, arc, best[np.clip(pt - 1, 0, N - 1)])
    df["near_next_best"] = _adjacent(net, best[np.clip(pt + 1, 0, N - 1)], arc)
    return df


def _adjacent(net: RoadNetwork, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 if arcs a and b are the same arc or share a junction node end-to-start (either order); else 0."""
    a = np.asarray(a); b = np.asarray(b)
    ok = (a >= 0) & (b >= 0)
    out = np.zeros(len(a))
    aa, bb = a[ok], b[ok]
    out[ok] = ((aa == bb) | (net.arcs[bb, 1] == net.arcs[aa, 0]) | (net.arcs[aa, 1] == net.arcs[bb, 0])).astype(float)
    return out
