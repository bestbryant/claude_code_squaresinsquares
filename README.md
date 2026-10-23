# Squares in Squares (Kaggle) — working repo

Competition: https://www.kaggle.com/competitions/squares-in-squares
Pack n unit squares (n = 1..1000) into the smallest axis-aligned square; score = sum over n of L/sqrt(n) - 1 (lower is better).

## Current state

* `submission.csv.gz` — ready to submit (gunzip first). **Official metric score: 22.374700578810** (verified locally with
  `metric.py`, an unmodified copy of the official v17 metric, after the pandas float64 round trip Kaggle performs).
* `solutions/nN.npy` — best verified packing per n (columns x, y, angle). Missing n fall back to the grid.

The score comes from merging every public notebook output (per n, smallest L that passes the official verifier on
float64-repr coordinates). Leaderboard context at the time of writing: #1 22.2504, per-puzzle best-of-all ~22.2647.
The remaining gap (~0.11) is almost entirely in n = K^2 + m with K >= 25, m ~ 10..30, where private families
(mostly team "shyjin") pack tilted-stick bands with very efficient band ends.

## Tools (`tools/`)

| file | purpose |
|---|---|
| `common.py` | load/write submissions, side length, official verification of float64 strings |
| `merge.py` | merge candidate CSVs into `solutions/` (keeps smallest verified L per n) |
| `make_submission.py` | write `submission.csv` and score it with the official metric |
| `compare.py` | per-n gap vs. the public per-puzzle leaderboard (`best_L.json` scraped from the per-puzzle notebook) |
| `slp.py` | sequential-LP polish (minimise L under linearised SAT constraints, trust region) |
| `relax.py` | overlap-penalty L-BFGS relaxation in a fixed box; insertion attempts |
| `bands.py` | constructive packings: tilted-stick bands, L-shape / pinwheel / lattice-corner layouts |
| `bandend.py`, `endsearch.py`, `polypack.py` | band-end gadget search (pack a trapezoidal band end optimally) |
| `slide.py` | gravity compaction (time-of-impact sliding) |
| `plot.py` | render packings |

## Findings / next steps

* Records in the upper part of each K band follow families with fixed fractional excess f = L - K and
  Delta = n - K^2 growing ~1 per K: two tilted-stick bands (top + right, sticks of h+1 squares across width h+f)
  around an upright core. Ideal (end-loss-free) L-shape gives Delta ≈ (f - sin t)(2K - h + f) + h f, i.e. ~30 at
  K=28, f=0.898, h=5 while the record is 25 and my constructions reach 20-21: **band-end losses are the whole game**.
* Optimised end gadgets (searched with `polypack`) recover only ~1 square per end over partial sticks so far.
* Ideas not yet finished: joint corner gadget where the two bands meet, thicker bands (h 7-10) with good ends,
  feeding new gadgets into the public fault-line composition DP.
