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
    "cos_heading_long", "heading_diff_long", "move_along", "win10_dist", "win10_dist_minus_min",
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


def adaptive_motion(pxy: np.ndarray, min_disp: float = 20.0, max_w: int = 15):
    """Displacement vector around each point using the smallest window (i-w..i+w)
    in which the vehicle has moved at least `min_disp` metres. Slow vehicles give
    noisy headings over short windows, so the window grows until movement is real."""
    N = len(pxy)
    idx = np.arange(N)
    dx = np.zeros(N); dy = np.zeros(N); done = np.zeros(N, bool)
    for w in range(1, max_w + 1):
        lo = np.clip(idx - w, 0, N - 1); hi = np.clip(idx + w, 0, N - 1)
        ddx = pxy[hi, 0] - pxy[lo, 0]; ddy = pxy[hi, 1] - pxy[lo, 1]
        upd = ~done
        dx[upd] = ddx[upd]; dy[upd] = ddy[upd]
        done |= np.hypot(ddx, ddy) >= min_disp
        if done.all():
            break
    return dx, dy


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

    # long-window (adaptive) direction: robust for slow vehicles
    ldx, ldy = adaptive_motion(pxy)
    ldisp = np.hypot(ldx, ldy)
    lvalid = ldisp[pt] >= 3.0
    hl = angle_diff(bearing_deg(ldx, ldy)[pt], net.bearing[arc])
    df["heading_diff_long"] = np.where(lvalid, hl, 90.0)
    df["cos_heading_long"] = np.where(lvalid, np.cos(np.radians(hl)), 0.0)
    ux = (net.bx[arc] - net.ax[arc]) / np.maximum(net.length[arc], 1e-6)
    uy = (net.by[arc] - net.ay[arc]) / np.maximum(net.length[arc], 1e-6)
    df["move_along"] = ldx[pt] * ux + ldy[pt] * uy           # signed metres moved along the arc direction
    acc = np.zeros(len(df))
    for o in range(-5, 6):
        j = np.clip(pt + o, 0, N - 1)
        d, _ = point_segment(pxy[j, 0], pxy[j, 1], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
        acc += np.minimum(d, 100.0)
    df["win10_dist"] = acc / 11
    df["win10_dist_minus_min"] = df["win10_dist"] - df.groupby("pt")["win10_dist"].transform("min")

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
