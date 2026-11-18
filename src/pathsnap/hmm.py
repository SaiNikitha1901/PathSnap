"""HMM + Viterbi route reconstruction.

Hidden state at observation i  = one of the candidate arcs of GPS point i.
Emission  log P(z_i | arc)      = alpha * log(max(p_norm, eps))
          p_norm = classifier probability, normalised over the point's candidates.
Transition log P(arc_j | arc_i) = -|route_dist - straight_dist| / (beta * max(1, dt))
          route_dist   = driving distance on the directed network (bounded Dijkstra)
          straight_dist = distance between the two GPS points
          unreachable pairs get a fixed big penalty instead of -inf, so a broken
          step never kills the whole chain (the decoder just "jumps").
          If the time gap is larger than gap_break seconds the chain restarts
          (all transitions equal).
Viterbi:  delta_i(b) = max_a [delta_{i-1}(a) + T(a,b)] + E_i(b), then back-track.

The expensive part (route distances) is computed once per trip by `precompute_steps`
so that decoder parameters can be tuned cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass
class HMMParams:
    alpha: float = 1.0
    beta: float = 10.0
    eps: float = 1e-4
    unreachable: float = -30.0
    gap_break: float = 60.0


@dataclass
class TripSteps:
    pts: np.ndarray            # GPS point index of each HMM step (points that have candidates)
    cand_arcs: list            # list of arrays: candidate arcs per step
    cand_rows: list            # list of arrays: row indices into the candidate dataframe
    route_d: list              # list of (K_prev, K_cur) arrays of route distances (step >= 1)
    gc: np.ndarray             # straight distance between consecutive steps
    dt: np.ndarray             # time gap between consecutive steps


def precompute_steps(df: pd.DataFrame, pxy: np.ndarray, times: np.ndarray, net: RoadNetwork) -> TripSteps:
    groups = df.groupby("pt", sort=True).indices
    pts = np.array(sorted(groups), dtype=np.int64)
    rows = [np.asarray(groups[p]) for p in pts]
    arcs = [df["arc"].to_numpy()[r] for r in rows]
    s_pos = [df["t"].to_numpy()[r] * net.length[a] for r, a in zip(rows, arcs)]
    gc = np.zeros(len(pts)); dt = np.zeros(len(pts))
    route_d = [None]
    for i in range(1, len(pts)):
        p, q = pts[i - 1], pts[i]
        gc[i] = float(np.hypot(*(pxy[q] - pxy[p])))
        dt[i] = float(times[q] - times[p])
        limit = max(400.0, 3.0 * gc[i] + 150.0)
        A, B = arcs[i - 1], arcs[i]
        sa, sb = s_pos[i - 1], s_pos[i]
        M = np.full((len(A), len(B)), np.inf)
        b_from = net.arcs[B, 0]
        for x, a in enumerate(A):
            dist, _ = net.dijkstra(int(net.arcs[a, 1]), limit)
            rest = net.length[a] - sa[x]
            for y in range(len(B)):
                if B[y] == a and sb[y] >= sa[x] - 15.0:
                    M[x, y] = abs(sb[y] - sa[x])
                    continue
                d = dist.get(int(b_from[y]))
                if d is not None:
                    M[x, y] = rest + d + sb[y]
        route_d.append(M)
    return TripSteps(pts, arcs, rows, route_d, gc, dt)


def viterbi(steps: TripSteps, probs: np.ndarray, params: HMMParams) -> np.ndarray:
    """Returns the decoded arc for every HMM step (same order as steps.pts)."""
    n = len(steps.pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    emis = []
    for r in steps.cand_rows:
        p = probs[r]
        p = p / max(p.sum(), 1e-12)
        emis.append(params.alpha * np.log(np.maximum(p, params.eps)))
    delta = emis[0].copy()
    back = [None]
    for i in range(1, n):
        M = steps.route_d[i]
        if steps.dt[i] > params.gap_break:
            T = np.zeros_like(M)
        else:
            beta = params.beta * max(1.0, steps.dt[i])
            T = np.where(np.isfinite(M), -np.abs(M - steps.gc[i]) / beta, params.unreachable)
        score = delta[:, None] + T
        bp = np.argmax(score, axis=0)
        delta = score[bp, np.arange(score.shape[1])] + emis[i]
        back.append(bp)
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    return np.array([steps.cand_arcs[i][path[i]] for i in range(n)], dtype=np.int64)


def classifier_top1(steps: TripSteps, probs: np.ndarray) -> np.ndarray:
    return np.array([a[np.argmax(probs[r])] for a, r in zip(steps.cand_arcs, steps.cand_rows)], dtype=np.int64)


def per_point(steps: TripSteps, decoded: np.ndarray, n_points: int) -> np.ndarray:
    """Expand step predictions to every GPS point (-1 where a point had no candidate)."""
    out = np.full(n_points, -1, dtype=np.int64)
    out[steps.pts] = decoded
    return out


def reconstruct_route(arc_seq: np.ndarray, net: RoadNetwork, fill: bool = True) -> list[int]:
    """Turn per-point arcs into a route: drop repeats and (optionally) fill the gap
    between consecutive different arcs with the shortest network path."""
    seq = [int(a) for a in arc_seq if a >= 0]
    route: list[int] = []
    for a in seq:
        if route and route[-1] == a:
            continue
        if route and fill:
            route.extend(net.path_arcs(route[-1], a))
        route.append(a)
    return route
