# Experiment protocol (shared by all experiments)

Read `CLAUDE_CODE_BRIEF.md` first for the goal and the known problems.

## Tools (already built, do not re-implement)
- `src/pefe_np.py`: batched numpy simulator. `BatchEnv` matches `environment3`
  exactly; `BatchBrain` is Brain3 in numpy; `Genome` is the sparse full-rank
  genome (`GenomePE(full_rank=True)` port, with the `clone()` and `hidden_max`
  bugs fixed). Run `python tests/test_equivalence.py` after touching it.
- `src/evolve.py`: evolution driver. Its defaults reproduce `evolve3_zero.py`;
  `python src/evolve.py -h` lists every knob. Each run writes `config.json`,
  `log.csv`, `best.npz` (best validation clone-team genome) and `final_pop.npz`.
- `src/evaluate.py --genome RUN/best.npz --json out.json` is the **standard
  evaluation**: a clone team of 10, 200 food, 1000 steps, 20 held-out seeds.
  It reports mean ± sd food per agent and a **blind** score (food sensors
  zeroed). A forager collapses when blind; a drifter doesn't.
  **Report final numbers only through this script.**
- `src/plot_runs.py --group label='glob' ... --out file.png` plots population
  mean, validation clone-team score and connection count over generations.
- `src/render.py --genome RUN/best.npz --out x.mp4` renders a video and a
  `_trails.png` trajectory image. You can view the PNG with the Read tool, or
  pull frames with `ffmpeg -i x.mp4 -vf fps=1 frame_%02d.png`.

## Baselines (standard protocol, results/baselines.json)
| controller | score | blind |
|---|---|---|
| zero output | 0.12 ± 0.12 | 0.12 |
| constant forward | 2.79 ± 0.54 | 2.79 |
| random dense net (18.8k conns, Brain3 defaults) | 7.71 ± 2.65 | 5.79 |
| random sparse net (50 conns) | 0.12 | 0.12 |
| hand-coded, stops when no food in view | 17.66 ± 1.33 | 0.12 |
| hand-coded, brief's rule (thrust to strongest sensor) | **19.70 ± 0.66** | 2.75 |

Reference evolution: `evolve.py` defaults, 3 seeds × 150 generations, logs in
`results/ref_default/` (see its `SUMMARY.md`).

## Rules
- Behavior must come from evolution. No hand-set weights in evolved genomes,
  and no gradients. A hand-wired Brain3 genome is fine as a sanity check of
  representability, but label it as such.
- Start from zero or very sparse genomes (`--init zeros` or `sparse:N`).
- **3 seeds minimum** per configuration you draw a conclusion from (0, 1, 2).
- Compare at **equal compute** where schemes differ in cost: use `--max-evals`
  (agent-episodes). One default generation is 120 agent-episodes, so 150
  generations is 18,000.
- **CPU: this machine has 4 cores shared by 3 experiment agents and the lead.
  Run at most ONE evolution process at a time (3 seeds one after another,
  or one seed per command).** A default generation takes about 1–1.5 s, and
  150 generations take about 3–4 min. Budget about 60–75 min wall-clock in total.
- Code changes: add new behaviour behind **new flags whose defaults keep
  the current behaviour**, so several experiments' changes merge cleanly.
  Keep changes small, put them in `src/pefe_np.py` / `src/evolve.py` (or a new
  file), and list them in your report.
- Put run outputs in `runs/` (gitignored). Copy what matters into
  `experiments/<X>/`: each run's `config.json`, `log.csv` and `best.npz`,
  the evaluate.py JSON, and plots. Keep it under ~5 MB, and don't commit videos
  larger than ~3 MB.
- Write `experiments/<X>/REPORT.md` with: what you changed and why, a table
  (config → per-seed standard score, blind score, connections, evals),
  fitness/connection plots, a short verdict, and the **exact recommended
  flags** for the combined run.
- Commit on your worktree branch; do not push, and do not touch `main`.
