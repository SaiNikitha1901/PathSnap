# Development experiments (student-test only)

These are the scripts behind the "How we pushed accuracy up" analysis and the "Other decoder and labelling experiments" table in the report. They use only the training trips and the student-test trips, never the evaluator trips.

Setup: `mkdir -p experiments/_cache`, then run from the repo root with `PATHSNAP_DATA="/path/to/Student folder"`.

| Script | What it tests | Result (student-test pointwise) |
|---|---|---|
| `probe.py` | Label distances and candidate miss rate for several radius/k values (chose 50 m, k = 10) | — |
| `exp1.py` | LightGBM (18 features) + HMM grid, plus an error breakdown (adjacent / reverse arcs) | 0.896 → 0.933 |
| `exp2.py` | GPS heading vs label direction on slow trips 77 and 7 | motivated the long-window heading |
| `exp3.py` | 23 features + signed same-arc transitions + U-turn penalty (writes `_cache/tr2.parquet`) | 0.952 |
| `exp4.py` | Error types after the fixes (in-route, reverse, adjacent) | — |
| `exp5_label_heading.py`, `exp6.py`, `exp9_label_heading_strong.py` | Heading term inside the label DP (weights 0, 6, 10, 20) and route-level F1 | 0.932–0.947 (kept weight 0) |
| `exp7.py` | Forward-backward posterior decoding vs Viterbi | 0.946 (Viterbi kept) |
| `exp8.py` | Second-stage stacked model on neighbour probabilities | 0.927 (dropped) |
