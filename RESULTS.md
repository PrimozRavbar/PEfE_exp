# PEfE: evolving foraging agents. Final results

Evolution from all-zero genomes now reliably produces agents that forage with
their food sensors. The final setup (the `src/evolve.py` defaults, called
"B apc" during development) scores **18.2–18.7 food per agent on all 3 seeds
with about 150 connections**. The hand-coded controller scores 19.7. The
original setup plateaus at about 12.4 and bloats to about 3,700 connections.

## What changed and why
- **numpy simulator (`src/pefe_np.py`)**: a port of Brain3, the sparse full-rank
  genome and `environment3`, batched over arenas and brains. torch could not be
  installed (download.pytorch.org is blocked here). The port reproduces
  `environment3` and the Brain3 equations (`tests/test_equivalence.py`), and the
  sparse batched forward pass is about 9× faster than the per-agent loop.
  Two bugs from the original genome code are fixed: `clone()` built a random
  genome first, and a `hidden_max` ≤ 0 after mutation silently killed a brain
  (hyperparameters are now clamped above 0).
- **Structural mutation (the main change)**: the original operator adds
  connections at the same per-entry rate it removes them. A sparse genome has far
  more empty entries than connections, so it bloats steadily. It also gives new
  connections weight sd 1, which Brain3's low input gain (0.01) makes nearly
  inert. The final operator adds a fixed expected **3 connections per child**,
  split 40% Win / 20% Wh / 40% Wout, with **weight sd 10**, and removes each
  existing connection with probability **0.02**. New connections are few but
  strong enough to change behaviour, and removal keeps the genome small.
- **Standard evaluation (`src/evaluate.py`)**: a clone team of 10, 200 food,
  1000 steps, 20 held-out seeds, plus a **blind** run with the 8 food sensors
  zeroed. A real forager collapses when blind; a drifter does not.
- Everything else keeps the `evolve3_zero.py` settings: generational,
  population 40, arenas of 10, 3 episodes per generation, top 8 survive
  (rank-weighted), 3 elites, default Brain3 gains, zero init. The full table is
  in `README.md`.

## Final numbers (standard evaluation, 20 held-out episodes)
| controller | score (mean ± sd) | blind | connections (Win/Wh/Wout) |
|---|---|---|---|
| **final setup, seed 0** | **18.48 ± 0.76** | 0.59 | 146 (66/33/47) |
| **final setup, seed 1** | **18.74 ± 0.91** | 0.87 | 149 (48/29/72) |
| **final setup, seed 2** | **18.15 ± 0.88** | 0.45 | 147 (62/25/60) |
| reference evolution (`evolve3_zero` defaults), seeds 0/1/2 | 12.00 / 12.26 / 12.87 | 9.76 / 1.80 / 4.63 | 2630 / 519 / 1308 |
| hand-coded, thrust toward strongest food sensor | 19.70 ± 0.66 | 2.75 | – |
| hand-coded, same but stops when nothing is in view | 17.66 ± 1.33 | 0.12 | – |
| random dense network | 7.71 ± 2.65 | 5.79 | 18,816 |
| always forward | 2.79 ± 0.54 | 2.79 | – |
| zero output | 0.12 ± 0.12 | 0.12 | – |

All runs: 150 generations, 18,000 agent-episodes, about 2 minutes each on one
core. The validation score reaches about 16 by generation 40 and 18–19 by
generation 100 (`results/final/curves.png`). The population mean connection
count stays at about 140–155. Runs, evaluation JSON, curves and a video of the
seed-1 team are in `results/final/`. Running `python src/evolve.py --seed 0`
with no other flags reproduces `results/final/s0/log.csv` row for row. The
seed-1 team's trajectories (`results/final/best_s1_trails.png`) show agents
steering from food to food. With food sensors zeroed they score under 1.

## What mattered, what didn't
- **Mattered most: few, strong new connections plus active removal.** With 3
  additions per child but sd 1, the score stays near 11 by generation 70. With
  sd 10 it reaches about 17. Removal at 0.02 holds genomes near 150 connections
  without a connection-cost term.
- **Higher input gain (`--add-sens` 1–10)** also raises fitness (best 19.06 on
  one seed). With the original structural rates, though, genomes bloat to about
  2,000 connections, and blind scores of 4–5 show partial drifting. It is not
  part of the final setup.
- **"Path" additions** (an input→hidden→output pair) give even smaller genomes
  (60–75 connections) at a slightly lower score (17.5–17.7).
- **Not helpful or inconclusive**: a connection-cost penalty alone, adding at a
  lower per-entry rate without the stronger scale, and changes to selection
  noise (fewer or more episodes, tournament, steady-state). These were tested
  on single seeds only.

## Where it still plateaus
Scores level off at about 18.5, between the hand-coded "stop" controller (17.7)
and the full hand-coded one (19.7). Brain3 has no bias term, so with no food in
view the hidden state decays to zero and the agent stops. The evolved agents
recover some exploration through recurrence, but they rarely search open space
efficiently. A bias or tonic drive in the brain, or combining the final setup
with a higher input gain, are the obvious next steps. Neither was run for this
final version.

## History
The experiment logs, screens and patches from development (experiments A
"signal gain", B "connection bloat", C "selection noise"; the B patch is now
part of `src/`) and the raw reference runs are in `archive/`. See
`archive/README.md`.
