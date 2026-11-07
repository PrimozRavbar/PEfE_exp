# Archive: raw state at the point the work was stopped (2026-11-07)

- `lead_runs/`: everything in the lead's `runs/`: the reference runs
  (`ref_default_s0..2`, including `final_pop.npz`), stdout logs, and a smoke test.
- `worktree_A|B|C/`: the experiment agents' complete `runs/` and `experiments/`
  folders, plus `code.patch`, their uncommitted code changes against commit
  e2e0e11. Apply a patch with `git apply archive/worktree_B/code.patch`.

`experiments/partial/` is the curated subset, with standard evaluations, a
summary and `RESULTS.md`. Start from there.
