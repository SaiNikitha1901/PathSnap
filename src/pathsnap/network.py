"""Road network of one trip: projected nodes, directed arcs, adjacency,
a spatial index for candidate search, and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
import shapely

from .geo import bearing_deg, project, utm_epsg


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, epsg: int | None = None):
        if epsg is None:
            c = nodes_lonlat.mean(axis=0)
            epsg = utm_epsg(float(c[0]), float(c[1]))
        self.epsg = epsg
        self.nodes_lonlat = nodes_lonlat
        self.xy = project(nodes_lonlat, epsg)
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        f, t = self.arcs[:, 0], self.arcs[:, 1]
        self.ax, self.ay = self.xy[f, 0], self.xy[f, 1]
        self.bx, self.by = self.xy[t, 0], self.xy[t, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)
        self.out_deg = np.bincount(f, minlength=self.n_nodes)
        self.in_deg = np.bincount(t, minlength=self.n_nodes)
        # CSR list of outgoing arcs per node
        order = np.argsort(f, kind="stable")
        self._out_arcs = order
        self._out_ptr = np.r_[0, np.cumsum(self.out_deg)]
        self._tree = None
        self._sp_cache: dict[int, tuple[float, dict, dict]] = {}

    # ---------- adjacency ----------
    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    # ---------- spatial index ----------
    @property
    def tree(self):
        if self._tree is None:
            lines = shapely.linestrings(
                np.stack([np.column_stack([self.ax, self.ay]),
                          np.column_stack([self.bx, self.by])], axis=1))
            self._tree = shapely.STRtree(lines)
        return self._tree

    # ---------- bounded Dijkstra (node -> node), cached ----------
    def dijkstra(self, src: int, limit: float):
        """Shortest network distances from node `src` to all nodes within `limit` metres.
        Returns (dist dict, predecessor-arc dict)."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for a in self.out_arcs(u):
                v = int(self.arcs[a, 1])
                nd = d + float(self.length[a])
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = int(a)
                    heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[src] = (limit, dist, pred)
        return dist, pred

    def route_distance(self, a: int, sa: float, b: int, sb: float, limit: float) -> float:
        """Driving distance from position sa metres along arc a to sb metres along arc b.
        Returns inf when b is not reachable within `limit`."""
        if a == b:
            if sb >= sa - 15.0:            # small backward jitter is tolerated
                return abs(sb - sa)
            # going backwards on the same arc means driving round a loop
        rest = self.length[a] - sa
        dist, _ = self.dijkstra(int(self.arcs[a, 1]), limit)
        d = dist.get(int(self.arcs[b, 0]))
        if d is None:
            return np.inf
        return float(rest + d + sb)

    def path_arcs(self, a: int, b: int, limit: float = 3000.0) -> list[int]:
        """Arcs strictly between arc a and arc b on the shortest path (empty if adjacent
        or unreachable)."""
        src, dst = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if src == dst:
            return []
        dist, pred = self.dijkstra(src, limit)
        if dst not in dist:
            return []
        out = []
        v = dst
        while v != src:
            e = pred[v]
            out.append(e)
            v = int(self.arcs[e, 0])
        return out[::-1]

    def connected(self, a: int, b: int, max_hops: int = 3) -> bool:
        """True if arc b equals arc a or can be reached from a within `max_hops` arcs."""
        if a == b:
            return True
        frontier = {int(self.arcs[a, 1])}
        target = int(self.arcs[b, 0])
        seen = set(frontier)
        for _ in range(max_hops):
            if target in frontier:
                return True
            nxt = set()
            for u in frontier:
                for e in self.out_arcs(u):
                    v = int(self.arcs[e, 1])
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
        return target in frontier

    # ---------- graph statistics (EDA) ----------
    def components(self):
        """Number of weakly / strongly connected components and isolated nodes."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(self.n_arcs), (self.arcs[:, 0], self.arcs[:, 1])),
                       shape=(self.n_nodes, self.n_nodes)).tocsr()
        n_weak, _ = connected_components(g, directed=True, connection="weak")
        n_strong, lab = connected_components(g, directed=True, connection="strong")
        isolated = int(((self.out_deg + self.in_deg) == 0).sum())
        largest_scc = int(np.bincount(lab).max()) if self.n_nodes else 0
        return {"weak_components": int(n_weak), "strong_components": int(n_strong),
                "largest_scc_frac": largest_scc / max(self.n_nodes, 1),
                "isolated_nodes": isolated}
