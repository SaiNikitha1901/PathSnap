"""Server-side inference pipeline.

Prediction path (no labels involved):
  raw files -> clean -> UTM -> candidates -> features -> model probabilities
           -> classifier-only top-1  and  HMM/Viterbi decoding -> route reconstruction
Evaluation path (server-side only, never sent to the UI):
  .route -> per-point reference labels (monotonic alignment) -> metrics
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import HMMParams, classifier_top1, per_point, precompute_steps, reconstruct_route, viterbi  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_to_route  # noqa: E402
from pathsnap.metrics import pointwise_accuracy, route_continuity, route_prf, route_length_error  # noqa: E402
from pathsnap.pipeline import DEFAULT_CFG, prepare_trip  # noqa: E402

ART = os.environ.get("ARTIFACTS_DIR", os.path.join(ROOT, "artifacts"))


class Model:
    def __init__(self, entry: dict):
        self.key, self.label, self.colour, self.type = entry["key"], entry["label"], entry["colour"], entry["type"]
        self.student_test = entry.get("student_test", {})
        path = os.path.join(ART, entry["file"])
        if self.type == "lightgbm_native":
            import lightgbm as lgb
            self._m = lgb.Booster(model_file=path)
        else:
            import joblib
            self._m = joblib.load(path)

    def proba(self, X: np.ndarray) -> np.ndarray:
        if self.type == "lightgbm_native":
            return self._m.predict(X)
        import pandas as pd
        return self._m.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]

    def public(self) -> dict:
        return {"key": self.key, "label": self.label, "colour": self.colour, "student_test": self.student_test}


class Ensemble:
    """Soft voting: the mean of the member models' probabilities."""
    def __init__(self, entry: dict, members: list):
        self.key, self.label, self.colour, self.type = entry["key"], entry["label"], entry["colour"], "ensemble"
        self.student_test = entry.get("student_test", {})
        self.members = members

    def proba(self, X):
        return np.mean([m.proba(X) for m in self.members], axis=0)

    def public(self) -> dict:
        return {"key": self.key, "label": self.label, "colour": self.colour, "student_test": self.student_test,
                "members": [m.key for m in self.members]}


class Engine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        reg = json.load(open(os.path.join(ART, "models.json")))
        self.models = {e["key"]: Model(e) for e in reg if e["type"] != "ensemble"}
        for e in reg:
            if e["type"] == "ensemble":
                self.models[e["key"]] = Ensemble(e, [self.models[k] for k in e["members"]])
        self.default_model = next((e["key"] for e in reg if e.get("default")), reg[0]["key"])
        fc = json.load(open(os.path.join(ART, "feature_config.json")))
        self.cfg = fc.get("cfg", DEFAULT_CFG)
        assert fc["features"] == FEATURES, "feature list in artifacts does not match code"
        self.hmm = HMMParams(**json.load(open(os.path.join(ART, "hmm_config.json"))))
        self._cache: dict = {}
        self._lock = threading.Lock()

    # ---------- trip listing (no labels) ----------
    def trips(self) -> list[dict]:
        out = []
        for tid in list_trips(self.data_dir):
            tr = np.loadtxt(os.path.join(self.data_dir, tid, f"{tid}.track"), ndmin=2)
            out.append({"trip_id": tid, "points": int(len(tr)), "duration_s": float(tr[-1, 2] - tr[0, 2]) if len(tr) else 0.0,
                        "bbox": [float(tr[:, 1].min()), float(tr[:, 0].min()), float(tr[:, 1].max()), float(tr[:, 0].max())]})
        return out

    # ---------- cached preparation ----------
    def _prepared(self, tid: str):
        with self._lock:
            if tid in self._cache:
                return self._cache[tid], True
        trip = load_trip(self.data_dir, tid, with_route=True)
        route = trip.route
        trip.route = None                                   # prediction pipeline never sees the route
        p = prepare_trip(trip, self.cfg, use_route_for_labels=False)
        steps = precompute_steps(p.df, p.pxy, p.times, p.net)
        ref = None
        if route is not None and len(route):                # evaluation only, kept on the server
            ref, _, _ = align_to_route(p.pxy, p.net, route, self.cfg["label_cap"], self.cfg.get("label_heading_weight", 0.0))
        X = p.df[FEATURES].to_numpy()
        item = {"p": p, "steps": steps, "X": X, "route": route, "ref": ref, "network": self._network(p)}
        with self._lock:
            self._cache[tid] = item
        return item, False

    def _network(self, p, radius: float = 70.0, cap: int = 5000):
        import shapely
        pts = shapely.points(p.pxy[:: max(1, len(p.pxy) // 1500)])
        near = np.unique(p.net.tree.query(pts, predicate="dwithin", distance=radius)[1])[:cap]
        ll = p.net.nodes_lonlat
        a = p.net.arcs[near]
        return [[round(ll[f, 1], 6), round(ll[f, 0], 6), round(ll[t, 1], 6), round(ll[t, 0], 6)] for f, t in a]

    # ---------- geometry ----------
    @staticmethod
    def _route_polylines(route: list[int], net) -> list[list[list[float]]]:
        """Arc sequence -> list of [lat, lon] polylines, split where consecutive arcs do not connect."""
        ll = net.nodes_lonlat
        lines, cur = [], []
        prev_to = None
        for a in route:
            f, t = int(net.arcs[a, 0]), int(net.arcs[a, 1])
            if prev_to is None or f != prev_to:
                if len(cur) > 1:
                    lines.append(cur)
                cur = [[round(ll[f, 1], 6), round(ll[f, 0], 6)]]
            cur.append([round(ll[t, 1], 6), round(ll[t, 0], 6)])
            prev_to = t
        if len(cur) > 1:
            lines.append(cur)
        return lines

    @staticmethod
    def _matched_points(pred: np.ndarray, p) -> list:
        """For every GPS point: [lat, lon] of its projection on the matched arc (None if unmatched)."""
        df = p.df
        key = dict(zip(zip(df["pt"].to_numpy(), df["arc"].to_numpy()), df["t"].to_numpy()))
        ll = p.net.nodes_lonlat
        out = []
        for i, a in enumerate(pred):
            if a < 0:
                out.append(None); continue
            t = key.get((i, int(a)), 0.5)
            f, e = p.net.arcs[a]
            lat = ll[f, 1] + t * (ll[e, 1] - ll[f, 1]); lon = ll[f, 0] + t * (ll[e, 0] - ll[f, 0])
            out.append([round(float(lat), 6), round(float(lon), 6), int(a)])
        return out

    # ---------- prediction ----------
    def predict(self, tid: str, model_keys: list[str]) -> dict:
        t0 = time.time()
        item, cached = self._prepared(tid)
        p, steps, X = item["p"], item["steps"], item["X"]
        prep_ms = (time.time() - t0) * 1000
        n = len(p.pxy)
        results = []
        for k in model_keys:
            m = self.models[k]
            t1 = time.time()
            prob = m.proba(X)
            clf = per_point(steps, classifier_top1(steps, prob), n)
            hmm = per_point(steps, viterbi(steps, prob, self.hmm), n)
            r_clf = reconstruct_route(clf, p.net, fill=False)
            r_hmm = reconstruct_route(hmm, p.net)
            infer_ms = (time.time() - t1) * 1000
            metrics = {"continuity_clf": route_continuity(clf, p.net), "continuity_hmm": route_continuity(hmm, p.net)}
            if item["ref"] is not None:
                metrics.update(accuracy_clf=pointwise_accuracy(clf, item["ref"]), accuracy_hmm=pointwise_accuracy(hmm, item["ref"]))
                prf = route_prf(r_hmm, item["route"], p.net)
                metrics.update({k2.replace("route_", "route_") : v for k2, v in prf.items()})
                metrics["route_f1_clf"] = route_prf(r_clf, item["route"], p.net)["route_f1"]
                le = route_length_error(r_hmm, item["route"], p.net)
                metrics.update(le)
                metrics["length_error_clf"] = route_length_error(r_clf, item["route"], p.net)["length_error"]
            metrics = {a: (round(float(b), 4) if b == b else None) for a, b in metrics.items()}
            results.append({"key": m.key, "label": m.label, "colour": m.colour, "infer_ms": round(infer_ms, 1),
                            "hmm_route": self._route_polylines(r_hmm, p.net), "clf_route": self._route_polylines(r_clf, p.net),
                            "matched_hmm": self._matched_points(hmm, p), "matched_clf": self._matched_points(clf, p),
                            "n_route_arcs": len(r_hmm), "metrics": metrics})
        gps = [[round(float(la), 6), round(float(lo), 6), float(t)] for lo, la, t in p.track]
        return {"trip_id": tid, "points": n, "candidate_rows": int(len(p.df)), "cached": cached,
                "prep_ms": round(prep_ms, 1), "total_ms": round((time.time() - t0) * 1000, 1),
                "hmm_params": self.hmm.__dict__, "gps": gps, "network": item["network"], "models": results}
