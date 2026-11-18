# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. This project snaps noisy 1 Hz GPS tracks onto a directed road network. A per-candidate classifier scores the nearby roads, then HMM + Viterbi decoding turns those scores into a connected route.

## Repository layout
| Folder | Contents |
|---|---|
| `src/pathsnap/` | data loading/cleaning (`io.py`), UTM projection (`geo.py`), road graph + Dijkstra (`network.py`), per-point label alignment (`labels.py`), candidate generation (`candidates.py`), features (`features.py`), HMM/Viterbi (`hmm.py`), metrics (`metrics.py`), Folium maps (`viz.py`) |
| `notebooks/` | `01_eda_and_models.ipynb`: EDA, candidates, 7-model comparison (Phase 1) |
| `results/` | CSV results, plots, Folium HTML maps |
| `artifacts/` | exported model + configs (Phase 2) |
| `server/` | FastAPI backend (Phase 2) |
| `ui/` | evaluator UI (Phase 2) |
| `docs/` | viva guide |

## Notebooks
| Notebook | What it does |
|---|---|
| `01_eda_and_models.ipynb` | Dataset inventory and quality checks, EDA plots, Folium map, per-point labels (monotonic DP alignment), candidate generation, 7-model comparison (GroupKFold by trip) |
| `02_bayesopt_hmm_export.ipynb` | Optuna TPE tuning of LightGBM, tuned vs untuned, HMM decoder tuning, classifier-only vs HMM, Folium route maps, export of `artifacts/` |
| `03_extended_models_ensemble.ipynb` | Extra models (HistGB, XGBoost, AdaBoost, calibrated linear SVM, deep MLP) + soft-voting ensemble |

Each notebook also has an `_executed` copy that keeps the outputs of our run.

## Two-laptop deployment (final evaluation)

**Laptop 1: API server (Nikitha's Mac)**
```
cd PathSnap
python3 -m venv .venv && .venv/bin/pip install -r server/requirements-server.txt
DATA_DIR="/path/to/evaluator_trips" .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
```
- `DATA_DIR` is a folder of trip sub-folders (`<id>/<id>.track/.nodes/.arcs/.route`). By default it is `data/eval_trips/`. For rehearsal, use `"Student folder/student_test"`.
- Get the Mac's IP with `ipconfig getifaddr en0`. The UI uses `http://<that-ip>:8000`. Check it with `curl http://<ip>:8000/health`.
- macOS firewall: when macOS asks, allow incoming connections for Python. You can also check System Settings → Network → Firewall → Options.
- If the college Wi-Fi isolates clients, connect both laptops to the same phone hotspot.
- The terminal prints one log block per request: trip, models, accuracy, latency. Show it to the evaluator.

**Laptop 2: UI (teammate)**
```
cd ui && python3 -m http.server 5500
```
Open `http://localhost:5500`, enter the API URL and click Connect. See `ui/README_TEAMMATE.md`.

**API endpoints:** `GET /health`, `GET /models`, `GET /trips` (ids, point counts and bounding boxes only), and `POST /predict {"trip_id": "...", "models": ["lightgbm", ...]}`. `/predict` returns the GPS points, the nearby road network, and for each model the HMM and classifier-only routes, the per-point matches and the server-side metrics. The reference route and the labels never leave the server.

## Running notebook 01
- **Colab:** upload the notebook, choose Run all, and upload `pathsnap_colab_bundle.zip` (holds `src/` + `Student folder/`) when asked.
- **Locally:**
  ```
  python -m venv .venv && .venv/bin/pip install numpy scipy pandas scikit-learn lightgbm shapely pyproj folium optuna matplotlib pyarrow jupyter
  PATHSNAP_BASE=$PWD PATHSNAP_DATA="/path/to/Student folder" jupyter nbconvert --execute --to notebook notebooks/01_eda_and_models.ipynb
  ```

The dataset is not committed. Evaluator-testing data is used only by the server at final evaluation.
