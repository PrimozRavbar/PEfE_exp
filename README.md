# PEfEvo

Experimenting with parameter-efficient evolution of recurrent neural networks embedded in simple 2D agents that interact with their environment and with one another.

The working pipeline is a numpy port of the "3" stack (Brain3, sparse full-rank genome, `environment3`) with a batched simulator. It evolves sparse foraging agents from all-zero genomes to about **18.5 food per agent with about 150 connections**. For comparison, the hand-coded controller scores 19.7. See `RESULTS.md`.

## Run locally

Needs Python 3 with `numpy`. Rendering also needs `matplotlib` and `ffmpeg`. No torch.

```bash
pip install numpy matplotlib
```

```bash
python src/evolve.py --out runs/x --seed 0
```

```bash
python src/evaluate.py --genome runs/x/best.npz --json runs/x/eval.json
```

```bash
python src/render.py --genome runs/x/best.npz --out runs/x/best.mp4
```

`evolve.py` writes `config.json` (every argument), `log.csv` (one row per generation), `best.npz` (the genome with the best validation score) and `final_pop.npz`. A 150-generation run takes about 2 minutes on one CPU core. `evaluate.py` prints the score over 20 held-out episodes (mean ± sd food per agent, clone team of 10), the **blind** score with food sensors zeroed (near zero for a real forager), and the connection count. `render.py` writes an MP4 and a `*_trails.png` trajectory plot. `python src/evaluate.py --baselines` reproduces `results/baselines.json`.

Tests:

```bash
python tests/test_equivalence.py
```

```bash
python tests/test_defaults.py
```

`test_equivalence.py` checks that the numpy simulator and brain match `environment3` and `Brain3`. `test_defaults.py` (about 1 minute) checks that the defaults are the final setup, and that a default 40-generation seed-0 run reproduces the reference `log.csv` in `tests/data/` and its evaluation. Both also run under `pytest`.

## Run in Colab

Open `PEfE_evolve_colab.ipynb` in Colab and run all cells. It clones the repository (`BRANCH = "main"`; set it to another branch to test that branch), evolves with `SEED` and `GENERATIONS`, evaluates (including the blind score), and shows the trajectory PNG and the video inline. No GPU needed.

## Default hyperparameters (`src/evolve.py`)

These defaults are the final "B apc" setup. `python src/evolve.py --out runs/x --seed 0` runs it with no extra flags. Every value is a flag, and each run saves its full set in `config.json`.

| flag | default | meaning |
|---|---|---|
| `--seed` | 0 | RNG seed |
| `--generations` | **150** | generations |
| `--max-evals` | 0 | stop after this many agent-episodes (0 = no limit) |
| `--population` | 40 | genomes per generation |
| `--arena-size` | 10 | agents per arena |
| `--episodes` | 3 | episodes per generation (fitness = mean food per agent) |
| `--survivors` | 8 | top genomes that reproduce (rank-weighted) |
| `--elites` | 3 | survivors copied unchanged |
| `--eval-mode` | mixed | `mixed`: shuffled arenas of different genomes; `clone`: teams of copies |
| `--init` | **zeros** | `zeros`, `random` (dense) or `sparse:N` |
| `--food` | 200 | food items per arena |
| `--steps` | 1000 | steps per episode |
| `--hidden-max` | 10000 | Brain3 hidden-state cap |
| `--add-hidden` | 0.5 | Brain3 recurrent gain |
| `--add-sens` | 0.01 | Brain3 input gain |
| `--leak-hidden` | 0.1 | Brain3 leak |
| `--max-epsilon` | 1.0 | Brain3 output noise scale |
| `--no-noise` | off | disable output noise |
| `--mutation-rate` | 0.1 | per-connection (and per-hyperparameter) perturbation probability |
| `--mutation-scale` | 0.05 | sd of weight perturbations |
| `--no-hyper-mutation` | off | freeze the per-genome hyperparameters |
| `--connection-rate` | 0.002 | per-entry add probability (used only with `--add-per-child none`) |
| `--add-per-child` | **3** | expected new connections per child |
| `--add-share` | **0.4,0.2,0.4** | split of the additions over Win, Wh, Wout |
| `--remove-rate` | **0.02** | per-connection removal probability |
| `--connection-scale` | **10** | sd of the weight of a new connection |
| `--no-caps` | off | drop the 20% per-matrix connection caps |
| `--add-path-prob` | 0.0 | per-child probability of adding an input→hidden→output path |
| `--connection-cost` | 0.0 | selection fitness = food − cost × connections |
| `--check-every` | 5 | validate the best genome every N generations |
| `--check-episodes` | 5 | validation episodes (seeds 500000+, disjoint from evaluation seeds) |
| `--checkpoint-every` | 0 | save the population every N generations (0 = off) |

Bold values differ from the original `evolve3_zero.py` setup. To run that setup instead, add `--add-per-child none --add-share none --remove-rate none --connection-scale 1`.

## Results summary

Standard evaluation: clone team of 10, 200 food, 1000 steps, 20 held-out episodes.

| controller | score | blind | connections |
|---|---|---|---|
| **evolved, default setup (3 seeds)** | **18.48 ± 0.76, 18.74 ± 0.91, 18.15 ± 0.88** | 0.59 / 0.87 / 0.45 | 146 / 149 / 147 |
| reference evolution (`evolve3_zero` defaults, 3 seeds) | 12.00, 12.26, 12.87 | 9.76 / 1.80 / 4.63 | 2630 / 519 / 1308 |
| hand-coded "thrust toward food" | 19.70 ± 0.66 | 2.75 | – |
| random dense network | 7.71 ± 2.65 | 5.79 | 18,816 |

Final runs, plots and video are in `results/final/`, baselines in `results/baselines.json`. Details are in `RESULTS.md`.

## Layout

- `src/pefe_np.py`: numpy Brain3, sparse genome, batched environment and brains.
- `src/evolve.py`: evolution driver. `src/evaluate.py`: standard evaluation and baselines. `src/render.py`: video and trajectory PNG. `src/plot_runs.py`: fitness curves over seeds.
- `tests/`: simulator equivalence and default-setup checks.
- `results/`: baselines and the final 3-seed runs.
- `archive/`: experiment logs, patches, screens and the raw runs from development, kept for reference. Nothing there is needed to run. See `archive/README.md`.
- Original torch code, unchanged: `environment.py`, `brain.py`, `agent.py`, `genome.py`, `run_episode.py` and `evolution.py` (first stack); `brain2.py`, `game_control.py`; and the "3" stack `brain3.py`, `genome_pe3.py`, `agent3.py`, `environment3.py`, `run_episode3.py`, `evolution3.py`, `evolve3_zero.py`. The numpy tests use `agent3.py` and `environment3.py`, which import only numpy.
- `PEfEvo_exp_1p0.ipynb`: the original experiment notebook.
