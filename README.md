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

## Running notebook 01
- **Colab:** upload the notebook, choose Run all, and upload `pathsnap_colab_bundle.zip` (holds `src/` + `Student folder/`) when asked.
- **Locally:**
  ```
  python -m venv .venv && .venv/bin/pip install numpy scipy pandas scikit-learn lightgbm shapely pyproj folium optuna matplotlib pyarrow jupyter
  PATHSNAP_BASE=$PWD PATHSNAP_DATA="/path/to/Student folder" jupyter nbconvert --execute --to notebook notebooks/01_eda_and_models.ipynb
  ```

The dataset is not committed. Evaluator-testing data is used only by the server at final evaluation.
