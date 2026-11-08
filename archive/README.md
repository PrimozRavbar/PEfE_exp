# Archive

Development material, kept for reference. Nothing here is needed to run,
evaluate or render. Nothing was deleted during the final tidy; it was moved here.

- `experiments/`: the curated experiment outputs, moved here from the old
  top-level `experiments/`. It contains `PROTOCOL.md`, `partial/SUMMARY.txt`
  (one line per run), and for each of A (signal gain), B (connection bloat) and
  C (selection noise) the `code.patch`, `runs/*/{config.json,log.csv,best.npz}`,
  screen scripts and logs. `partial/B/code.patch` is now applied in `src/`.
  The three `B_apc_s*` runs are copied to `results/final/`.
- `results/ref_default/`: the reference runs (`evolve3_zero` defaults, 3 seeds)
  with their evaluation and summary. `results/handwired_sanity.txt`: the
  hand-wired Brain3 check.
- `scripts/handwired.py` (hand-wired Brain3 sanity check) and
  `scripts/bench_speed.py` (batched vs per-agent speed). Both run from the repo
  root, for example `python archive/scripts/handwired.py`.
- `CLAUDE_CODE_BRIEF.md`: the original task brief.
- `lead_runs/`: the raw `runs/` of the reference runs (including
  `final_pop.npz`), stdout logs and a smoke test.
- `worktree_A|B|C/`: the experiment agents' complete raw `runs/` and
  `experiments/` folders, plus `code.patch` (their code changes against commit
  e2e0e11).
