"""Geometry helpers. Lon/lat are angles, so we project to UTM (metres) before
measuring any distance. The UTM zone is chosen from the trip's mean longitude."""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

_CACHE: dict[int, Transformer] = {}


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def get_transformer(epsg: int) -> Transformer:
    if epsg not in _CACHE:
        _CACHE[epsg] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return _CACHE[epsg]


def project(lonlat: np.ndarray, epsg: int) -> np.ndarray:
    if len(lonlat) == 0:
        return np.zeros((0, 2))
    x, y = get_transformer(epsg).transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB.
    Returns (distance, t) where t in [0,1] is the projection fraction along AB."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t


def bearing_deg(dx, dy):
    """Compass-style bearing of a vector in projected coords (0 = north, clockwise)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in degrees [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
