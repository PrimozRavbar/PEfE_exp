# PEfE: evolving foraging agents. Status: STOPPED EARLY (partial results)

The owner stopped the work partway, to save usage. The shared infrastructure
and reference runs are complete. Experiments A (signal gain), B (connection
bloat) and C (selection noise) were interrupted mid-way. Experiment D, the
combined run and the independent review were **not done**. All partial outputs
are preserved in `experiments/partial/`.

## What was built (complete, tested)
- `src/pefe_np.py`: numpy port of Brain3 and the sparse full-rank genome. torch
  could not be installed: download.pytorch.org is blocked by the network policy.
  The batched environment reproduces `environment3` exactly
  (`tests/test_equivalence.py`). Batched brains use a sparse matvec, about 9×
  faster than the per-agent loop (`src/bench_speed.py`). Fixed: `clone()` built a
  random genome first; `hidden_max <= 0` killed brains (hyperparameters now
  clamped > 0).
- `src/evaluate.py`: the standard protocol. A clone team of 10, 200 food, 1000
  steps, 20 held-out seeds, plus a **blind** check (food sensors zeroed): foragers
  collapse when blind, drifters don't.
- `src/evolve.py`: configurable driver. Its defaults reproduce `evolve3_zero.py`.
- `src/render.py` (MP4 and trajectory PNG), `src/plot_runs.py`,
  `src/handwired.py`, and `PEfE_evolve_colab.ipynb`.
  (The notebook expects `results/final/FLAGS.txt`, which was never written
  because the combine step was not run. Pass the B flags below by hand.)

## Baselines (standard protocol, `results/baselines.json`)
| controller | score | blind |
|---|---|---|
| zero output | 0.12 ± 0.12 | 0.12 |
| random dense net (18.8k conns) | 7.71 ± 2.65 | 5.79 |
| hand-coded, stops when nothing is in view | 17.66 ± 1.33 | 0.12 |
| hand-coded, brief's rule | **19.70 ± 0.66** | 2.75 |
| reference evolution (`evolve3_zero` defaults, 150 gen, 3 seeds) | 12.0 / 12.3 / 12.9 | 9.8 / 1.8 / 4.6 |

The reference plateaus at about 12.4 by generation 20, and connections bloat
linearly to about 3,700.

## Partial results (standard eval, 20 held-out episodes; complete 150-gen runs only)
| experiment / config | seed scores | blind | connections |
|---|---|---|---|
| **B "apc"**: 3 added connections per child (40% Win / 20% Wh / 40% Wout), remove 0.02, connection scale 10 | 18.48 ± 0.76, 18.74 ± 0.91, 18.15 ± 0.88 | 0.6 / 0.9 / 0.5 | 146 / 149 / 147 |
| B "path": add 1e-4, remove 0.05, connection scale 10 | 17.67, 17.56, 17.46 | about 0.2 | 75 / 70 / 60 |
| B "path" + add_sens 0.1 (2 seeds finished) | 17.84 | 0.6 | 68 |
| A: add_sens 10 (2 of 3 seeds finished) | 19.06 ± 0.69, 16.76 ± 1.53 | 3.9 / 5.0 | 1729 / 1994 |

Tentative conclusions (not yet confirmed by a combined run or a review):
- **The biggest lever was making new connections strong and few** (connection
  scale 10, a handful of additions per child, active removal). Fitness rose from
  the reference's ~12.4 to ~18.5 with ~150 connections (~25× fewer), on all 3
  seeds. Near-zero blind scores show the agents use their food sensors rather
  than drifting.
- Higher input gain (A) also lifts fitness, but with default structural rates the
  genomes still bloat to ~2k connections, and blind scores of 4–5 show some
  drift.
- C (selection noise) was still screening; its single-seed screens are in
  `experiments/partial/C/` and inconclusive.
- Brain3 has no bias, so without food in view an agent stops. This caps
  sensor-driven policies near the hand-coded-stop score (17.7). The best
  B genomes slightly exceed that, so some exploration has evolved.

## Where to pick up
1. Apply `experiments/partial/B/code.patch` (adds `--add-per-child`,
   `--add-share`, `--add-path-prob`) and, if wanted, `experiments/partial/A/code.patch`.
2. Combined run: B "apc" flags plus A's `--add-sens 10` (or 1.0), 3 seeds × 300+
   generations.
3. Render the best genome (`src/render.py`) and check that the blind score
   stays near zero.

Layout of `experiments/partial/<A|B|C>/`: `code.patch` (uncommitted code
changes from that agent's worktree), `runs/*/{config.json,log.csv,best.npz}`,
screen logs and scripts, and `SUMMARY.txt` (per-run generations, best validation
and connections). Interrupted runs have shorter logs.
