"""Length-error Folium maps (correct / wrongly added / missed arcs) for student-test trips.
Usage: .venv/bin/python scripts/make_length_error_maps.py 00000009 00000056"""
import os
import sys, json, joblib, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "server")
from pipeline import Engine  # server engine: same models + HMM as the API
from pathsnap.io import load_trip
from pathsnap.hmm import per_point, viterbi, reconstruct_route, classifier_top1
from pathsnap.viz import length_error_map
from pathsnap.metrics import route_length_error
eng = Engine(os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test")
for tid in sys.argv[1:]:
    item, _ = eng._prepared(tid); p, st, X = item["p"], item["steps"], item["X"]
    prob = eng.models["ensemble"].proba(X)
    for mode in ["hmm", "clf"]:
        pred = per_point(st, viterbi(st, prob, eng.hmm) if mode == "hmm" else classifier_top1(st, prob), len(p.pxy))
        route = reconstruct_route(pred, p.net, fill=(mode == "hmm"))
        le = route_length_error(route, item["route"], p.net)
        print(tid, mode, {k: round(v, 3) for k, v in le.items()})
        length_error_map(p.track, p.net, route, item["route"], f"Trip {tid} · Ensemble {'+ HMM' if mode == 'hmm' else 'classifier only'}").save(f"results/maps/length_error_{tid}_{mode}.html")
