"""Candidate generation: for every GPS point, the road arcs within `radius` metres
(at most `k` nearest). Each (point, candidate arc) pair becomes one row for the classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import point_segment
from .network import RoadNetwork


def generate_candidates(pxy: np.ndarray, net: RoadNetwork, radius: float = 50.0, k: int = 8) -> pd.DataFrame:
    import shapely
    pts = shapely.points(pxy)
    pi, ai = net.tree.query(pts, predicate="dwithin", distance=radius)
    if len(pi) == 0:
        return pd.DataFrame(columns=["pt", "arc", "dist", "t"])
    d, t = point_segment(pxy[pi, 0], pxy[pi, 1], net.ax[ai], net.ay[ai], net.bx[ai], net.by[ai])
    df = pd.DataFrame({"pt": pi.astype(np.int64), "arc": ai.astype(np.int64), "dist": d, "t": t})
    df = df.sort_values(["pt", "dist"], kind="stable")
    df["dist_rank"] = df.groupby("pt").cumcount()
    df = df[df["dist_rank"] < k].reset_index(drop=True)
    return df
