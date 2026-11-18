"""End-to-end preparation of one trip: clean -> project -> labels -> candidates -> features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .features import build_features
from .geo import project
from .io import Trip, clean_track
from .labels import align_to_route
from .network import RoadNetwork

DEFAULT_CFG = {"radius": 50.0, "k": 10, "window": 2, "label_cap": 100.0, "label_heading_weight": 0.0}


@dataclass
class PreparedTrip:
    trip_id: str
    net: RoadNetwork
    track: np.ndarray       # cleaned lon, lat, t
    pxy: np.ndarray
    times: np.ndarray
    df: pd.DataFrame        # candidate rows with features (+ label if route given)
    ref: np.ndarray | None  # reference arc per point (or None)
    ref_dist: np.ndarray | None
    clean_report: dict = field(default_factory=dict)


def prepare_trip(trip: Trip, cfg: dict = DEFAULT_CFG, use_route_for_labels: bool = True) -> PreparedTrip:
    track, rep = clean_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs)
    pxy = project(track[:, :2], net.epsg)
    times = track[:, 2]
    cands = generate_candidates(pxy, net, cfg["radius"], cfg["k"])
    df = build_features(cands, pxy, times, net, cfg["window"])
    df["trip"] = trip.trip_id
    ref = ref_dist = None
    if use_route_for_labels and trip.route is not None and len(trip.route):
        ref, _, ref_dist = align_to_route(pxy, net, trip.route, cfg["label_cap"], cfg.get("label_heading_weight", 0.0))
        df["label"] = (df["arc"].to_numpy() == ref[df["pt"].to_numpy()]).astype(int)
    return PreparedTrip(trip.trip_id, net, track, pxy, times, df, ref, ref_dist, rep)
