"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000
Env:  DATA_DIR=<folder of trip sub-folders>  (default data/eval_trips)
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import ROOT, Engine

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data", "eval_trips"))
log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | PATHSNAP | %(message)s", datefmt="%H:%M:%S")

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = Engine(DATA_DIR)
log.info("loaded models: %s | data dir: %s | trips: %d | HMM %s", ", ".join(engine.models), DATA_DIR, len(engine.trips()), engine.hmm.__dict__)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client, resp.status_code, (time.time() - t0) * 1000)
    return resp


class PredictRequest(BaseModel):
    trip_id: str
    models: list[str] = Field(default_factory=lambda: ["lightgbm"])


@app.get("/health")
def health():
    return {"status": "ok", "models": list(engine.models), "trips": len(engine.trips())}


@app.get("/models")
def models():
    return [dict(m.public(), default=(k == engine.default_model)) for k, m in engine.models.items()]


@app.get("/trips")
def trips():
    return engine.trips()


@app.post("/predict")
def predict(req: PredictRequest):
    ids = {t["trip_id"] for t in engine.trips()}
    if req.trip_id not in ids:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    bad = [k for k in req.models if k not in engine.models]
    if bad or not req.models:
        raise HTTPException(400, f"unknown models {bad}")
    log.info("PREDICT trip=%s models=%s", req.trip_id, req.models)
    out = engine.predict(req.trip_id, req.models)
    for m in out["models"]:
        mt = m["metrics"]
        log.info("   %-16s acc clf=%s hmm=%s | route F1=%s | continuity=%s | %d route arcs | %.0f ms",
                 m["key"], mt.get("accuracy_clf"), mt.get("accuracy_hmm"), mt.get("route_f1"), mt.get("continuity_hmm"), m["n_route_arcs"], m["infer_ms"])
    log.info("   trip %s: %d GPS points, %d candidate rows, prep %.0f ms (cached=%s), total %.0f ms",
             req.trip_id, out["points"], out["candidate_rows"], out["prep_ms"], out["cached"], out["total_ms"])
    return out
