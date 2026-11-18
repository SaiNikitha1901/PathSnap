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
