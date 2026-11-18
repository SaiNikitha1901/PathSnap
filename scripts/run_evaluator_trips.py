"""Final evaluation run: every evaluator trip through the same Engine the API uses, with the frozen
artifacts. Writes results/evaluator_run.csv. Run ONCE after all model and decoder choices are frozen;
these numbers must not be used to choose anything.
Usage: DATA_DIR=data/eval_trips .venv/bin/python scripts/run_evaluator_trips.py"""
import os, sys
import pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src")); sys.path.insert(0, os.path.join(BASE, "server"))
from pipeline import Engine  # noqa: E402

eng = Engine(os.environ.get("DATA_DIR", os.path.join(BASE, "data", "eval_trips")))
models = sys.argv[1:] or ["ensemble", "xgboost", "lightgbm", "random_forest", "mlp"]
rows = []
for t in eng.trips():
    out = eng.predict(t["trip_id"], models)
    for m in out["models"]:
        rows.append({"trip": t["trip_id"], "points": out["points"], "model": m["key"], **m["metrics"], "infer_ms": m["infer_ms"]})
        print(t["trip_id"], m["key"], m["metrics"].get("accuracy_hmm"), m["metrics"].get("length_error"))
df = pd.DataFrame(rows); df.to_csv(os.path.join(BASE, "results", "evaluator_run.csv"), index=False)
w = df.groupby("model").apply(lambda g: pd.Series({c: (g[c] * g.points).sum() / g.points.sum() for c in ["accuracy_clf", "accuracy_hmm", "route_f1", "length_error", "continuity_hmm"]}))
w.to_csv(os.path.join(BASE, "results", "evaluator_summary.csv")); print(w.round(4))
