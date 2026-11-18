"""Reading the raw trip files.

Each trip folder <id>/ holds:
  <id>.track  lon lat t        (GPS observations, ~1 Hz)
  <id>.nodes  lon lat          (row index = node id)
  <id>.arcs   from_node to_node (row index = arc id, directed)
  <id>.route  arc_id           (reference route, chronological; may be absent)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (M, 2) lon, lat
    arcs: np.ndarray             # (A, 2) int from, to
    route: Optional[np.ndarray]  # (R,) int arc ids, or None


def _load(path: str, dtype, ncols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, ncols), dtype=dtype)
    return arr


def list_trips(root: str) -> list[str]:
    """Trip ids = sub-folders of `root` that contain a .track file."""
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, name, f"{name}.track")):
            out.append(name)
    return out


def load_trip(root: str, trip_id: str, with_route: bool = True) -> Trip:
    d = os.path.join(root, trip_id)
    track = _load(os.path.join(d, f"{trip_id}.track"), float, 3)
    nodes = _load(os.path.join(d, f"{trip_id}.nodes"), float, 2)
    arcs = _load(os.path.join(d, f"{trip_id}.arcs"), np.int64, 2)
    route = None
    rp = os.path.join(d, f"{trip_id}.route")
    if with_route and os.path.isfile(rp):
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    return Trip(trip_id, track, nodes, arcs, route)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Basic cleaning: drop non-finite / out-of-range rows, sort by time,
    drop duplicate timestamps. Returns the cleaned track and a report."""
    rep = {"raw_points": int(len(track))}
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    rep["invalid_rows"] = int((~ok).sum())
    tr = track[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    rep["out_of_order"] = int((order != np.arange(len(order))).sum())
    tr = tr[order]
    if len(tr) > 1:
        keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    else:
        keep = np.ones(len(tr), bool)
    rep["duplicate_timestamps"] = int((~keep).sum())
    tr = tr[keep]
    rep["clean_points"] = int(len(tr))
    return tr, rep
