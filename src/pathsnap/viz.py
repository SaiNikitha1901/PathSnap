"""Folium maps: raw GPS, local road network, reference route, predicted routes."""
from __future__ import annotations

import folium
import numpy as np


def _arc_lines(net, arcs):
    return [[(net.nodes_lonlat[net.arcs[a, 0], 1], net.nodes_lonlat[net.arcs[a, 0], 0]),
             (net.nodes_lonlat[net.arcs[a, 1], 1], net.nodes_lonlat[net.arcs[a, 1], 0])] for a in arcs]


def trip_map(track, net, ref_route=None, routes: dict | None = None, network_radius_m: float = 80.0,
             max_network_arcs: int = 6000):
    lat, lon = track[:, 1], track[:, 0]
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=14, tiles="OpenStreetMap")
    # road network near the trajectory
    import shapely
    from .geo import project
    pxy = project(track[:, :2], net.epsg)
    near = np.unique(net.tree.query(shapely.points(pxy[:: max(1, len(pxy) // 2000)]),
                                    predicate="dwithin", distance=network_radius_m)[1])[:max_network_arcs]
    g = folium.FeatureGroup(name=f"Road network (arcs within {int(network_radius_m)} m)")
    for line in _arc_lines(net, near):
        folium.PolyLine(line, color="#9a9a9a", weight=1.5, opacity=0.6).add_to(g)
    g.add_to(m)
    if ref_route is not None:
        g = folium.FeatureGroup(name="Reference route (ground truth)")
        for line in _arc_lines(net, ref_route):
            folium.PolyLine(line, color="#2ca02c", weight=6, opacity=0.7).add_to(g)
        g.add_to(m)
    colors = ["#ff7f0e", "#d62728", "#9467bd", "#1f77b4"]
    for (name, arcs), c in zip((routes or {}).items(), colors):
        g = folium.FeatureGroup(name=name)
        for line in _arc_lines(net, arcs):
            folium.PolyLine(line, color=c, weight=3, opacity=0.9, dash_array="6" if "classifier" in name.lower() else None).add_to(g)
        g.add_to(m)
    g = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(track) // 1500)
    for la, lo in zip(lat[::step], lon[::step]):
        folium.CircleMarker((la, lo), radius=2, color="#1f4fd1", fill=True, fill_opacity=0.8, weight=0).add_to(g)
    g.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    return m


def length_error_map(track, net, pred_route, ref_route, title: str = ""):
    """Folium map that shows WHERE the length error comes from:
    teal = predicted arcs that are also in the reference route (correct),
    red = predicted arcs not in the reference (wrongly added, d+),
    dashed amber = reference arcs the prediction missed (d-).
    Used offline for the report; the UI never receives reference geometry."""
    P, R = set(int(a) for a in pred_route), set(int(a) for a in ref_route)
    lat, lon = track[:, 1], track[:, 0]
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=14, tiles="OpenStreetMap")
    g = folium.FeatureGroup(name="Raw GPS (faint)")
    step = max(1, len(track) // 1200)
    for la, lo in zip(lat[::step], lon[::step]):
        folium.CircleMarker((la, lo), radius=1.5, color="#555", fill=True, fill_opacity=0.35, weight=0).add_to(g)
    g.add_to(m)
    L = lambda s: float(net.length[list(s)].sum()) if s else 0.0
    for name, arcs, col, dash, wt in [("Correct (in both routes)", P & R, "#1b9e77", None, 5),
                                      ("Missed (d-)", R - P, "#f0a202", "8 6", 7),
                                      ("Wrongly added (d+)", P - R, "#d1173b", None, 7)]:
        g = folium.FeatureGroup(name=f"{name}: {L(arcs):,.0f} m")
        for line in _arc_lines(net, sorted(arcs)):
            folium.PolyLine(line, color=col, weight=wt, opacity=0.95, dash_array=dash).add_to(g)
        g.add_to(m)
    d0 = L(R) or 1.0
    err = (L(R - P) + L(P - R)) / d0
    html = (f"<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 14px;border-radius:10px;"
            f"box-shadow:0 2px 10px rgba(0,0,0,.25);font:13px Arial;line-height:1.5'><b>{title}</b><br>"
            f"Length error = (added {L(P - R):,.0f} m + missed {L(R - P):,.0f} m) / true {d0 / 1000:.2f} km = <b>{100 * err:.1f}%</b><br>"
            f"<span style='color:#1b9e77'>&#9644;</span> correct &nbsp; <span style='color:#d1173b'>&#9644;</span> wrongly added &nbsp; "
            f"<span style='color:#f0a202'>&#9644;</span> missed (dashed)</div>")
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    return m
