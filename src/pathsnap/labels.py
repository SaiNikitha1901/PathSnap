"""Per-observation reference labels.

The .route file lists the arcs the vehicle drove, in order, but it does not say
which GPS point belongs to which arc. We derive a label for each point with a
monotonic alignment (dynamic programming):

    point i is assigned route position k(i), with k(0) <= k(1) <= ... (never goes back)
    and the total (capped) point-to-arc distance is minimised.

The labels are derived, so they are reference information, not perfect truth.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .network import RoadNetwork


def align_to_route(pxy: np.ndarray, net: RoadNetwork, route: np.ndarray, cap: float = 100.0,
                   heading_weight: float = 20.0):
    """Returns (labels arc id per point, route position per point, distance to label arc).

    cost(point i, route arc k) = min(distance, cap) + heading_weight * (1 - cos(angle)) / 2
    The heading term (0 when the vehicle moves along the arc direction, heading_weight
    when it moves exactly against it) matters when a route uses the same road in both
    directions (out-and-back trips): distance alone cannot tell the two arcs apart."""
    N, R = len(pxy), len(route)
    if N == 0 or R == 0:
        return np.full(N, -1), np.full(N, -1), np.full(N, np.nan)
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    from .features import adaptive_motion
    mdx, mdy = adaptive_motion(pxy)
    mlen = np.hypot(mdx, mdy)
    ux = (bx - ax) / np.maximum(net.length[route], 1e-6); uy = (by - ay) / np.maximum(net.length[route], 1e-6)

    def hcost(i):
        if heading_weight <= 0 or mlen[i] < 15.0:
            return 0.0
        cos = (mdx[i] * ux + mdy[i] * uy) / mlen[i]
        return heading_weight * (1 - cos) / 2
    back = np.zeros((N, R), dtype=np.int32)
    idx = np.arange(R)
    d0, _ = point_segment(pxy[0, 0], pxy[0, 1], ax, ay, bx, by)
    cost = np.minimum(d0, cap) + hcost(0)
    for i in range(1, N):
        cm = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost == cm, idx, 0))
        back[i] = arg
        di, _ = point_segment(pxy[i, 0], pxy[i, 1], ax, ay, bx, by)
        cost = cm + np.minimum(di, cap) + hcost(i)
    pos = np.empty(N, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    lab = route[pos]
    dist, _ = point_segment(pxy[:, 0], pxy[:, 1], net.ax[lab], net.ay[lab], net.bx[lab], net.by[lab])
    return lab, pos, dist
