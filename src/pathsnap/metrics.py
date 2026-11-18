"""Evaluation metrics for map matching."""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def pointwise_accuracy(pred: np.ndarray, ref: np.ndarray) -> float:
    m = ref >= 0
    return float((pred[m] == ref[m]).mean()) if m.any() else float("nan")


def candidate_miss_rate(df, ref: np.ndarray) -> float:
    """Share of GPS points whose reference arc is not among their candidates
    (points with no candidate at all also count as misses)."""
    hit = np.zeros(len(ref), bool)
    sub = df[df["arc"].to_numpy() == ref[df["pt"].to_numpy()]]
    hit[sub["pt"].to_numpy()] = True
    return float(1 - hit[ref >= 0].mean())


def route_continuity(pred_points: np.ndarray, net: RoadNetwork, max_hops: int = 3) -> float:
    """Share of consecutive predicted arcs that are the same arc or reachable within
    `max_hops` arcs in the driving direction. 1.0 = a fully connected route."""
    p = pred_points[pred_points >= 0]
    if len(p) < 2:
        return float("nan")
    ok = 0; n = 0
    cache = {}
    for a, b in zip(p[:-1], p[1:]):
        key = (int(a), int(b))
        if key not in cache:
            cache[key] = net.connected(int(a), int(b), max_hops)
        ok += cache[key]; n += 1
    return ok / n


def route_prf(pred_route, ref_route, net: RoadNetwork) -> dict:
    """Length-weighted precision / recall / F1 between the predicted arc set and the
    reference route arc set (directed arcs)."""
    P, R = set(int(a) for a in pred_route), set(int(a) for a in ref_route)
    if not P or not R:
        return {"route_precision": 0.0, "route_recall": 0.0, "route_f1": 0.0}
    inter = P & R
    li = net.length[list(inter)].sum() if inter else 0.0
    prec = li / max(net.length[list(P)].sum(), 1e-9)
    rec = li / max(net.length[list(R)].sum(), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"route_precision": float(prec), "route_recall": float(rec), "route_f1": float(f1)}
