import os
import sys, time, numpy as np
sys.path.insert(0, "src")
from pathsnap.io import load_trip, clean_track, list_trips
from pathsnap.network import RoadNetwork
from pathsnap.geo import project
from pathsnap.labels import align_to_route
from pathsnap.candidates import generate_candidates
root = os.environ.get("PATHSNAP_DATA", "Student folder") + "/train"
ids = list_trips(root)[:: 8]
alld=[]; 
for tid in ids:
    t0=time.time(); tr = load_trip(root, tid); trk,_ = clean_track(tr.track)
    net = RoadNetwork(tr.nodes, tr.arcs); pxy = project(trk[:, :2], net.epsg)
    lab, pos, d = align_to_route(pxy, net, tr.route)
    c = generate_candidates(pxy, net, 60, 30)
    m = c.merge(dict_ := __import__("pandas").DataFrame({"pt":np.arange(len(lab)),"lab":lab}), on="pt")
    hit = m[m.arc==m.lab]
    rank = np.full(len(lab), 99); rank[hit.pt.values] = hit.dist_rank.values
    alld.append(d)
    print(tid, net.epsg, len(trk), len(tr.route), "covered route pos frac %.2f"%(len(np.unique(pos))/len(tr.route)),
      "labdist p50 %.1f p95 %.1f p99 %.1f"%tuple(np.percentile(d,[50,95,99])),
      "miss@k4 %.3f k8 %.3f k12 %.3f k30 %.3f"%tuple((rank>=k).mean() for k in [4,8,12,30]),
      "ncand %.1f"%c.groupby('pt').size().mean(), "%.1fs"%(time.time()-t0))
d=np.concatenate(alld); print("all labdist pct", np.percentile(d,[50,90,95,99,99.5]))
