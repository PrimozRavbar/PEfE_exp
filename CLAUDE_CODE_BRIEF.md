# PEfE — Claude Code brief: evolve agents that actually behave

Repo: https://github.com/PrimozRavbar/PEfE_exp (public). The current code is the "3" stack in `src/`:
`brain3.py`, `genome_pe3.py`, `agent3.py`, `environment3.py`, `run_episode3.py`, `evolution3.py`.
A starting driver, `evolve3_zero.py`, comes with this brief (copy it into `src/` if useful).

## Goal
Build a working project that **evolves recurrent-network agents that visibly behave in an environment**: they seek out and collect food, and that behavior emerges through evolution. Fitness should rise from near zero and keep improving rather than stall. Maximize it; there is no fixed target number.

Start from **very sparse connectivity** (all-zero weights or a handful of connections) and keep genomes sparse. The project is about parameter-efficient evolution, so report connection counts alongside fitness.

The owner does not want to babysit this. Work autonomously, make your own decisions, and report results at the end.

## Freedom
You can change anything:
- Fix or rewrite any existing file (brain, genome, mutation operators, selection, environment, episode runner).
- Change any hyperparameters: Brain3 gains and leaks, mutation rates and scales, population size, episode length, food count.
- Create new files, a new brain, or an entirely new environment if the current one gets in the way.
- Use whatever evolutionary scheme works: generational, steady-state, NEAT-style growth, novelty bonus, curriculum, and so on.

The one rule: behavior must come from evolution, not hand-set weights or gradient training. Hand-coded controllers are fine as baselines and sanity checks. Whenever you change a component, say what changed and why.

## How to judge it
- Fitness (food collected per agent) climbs reliably across at least 3 seeds.
- The best agents visibly forage. Render a short video and check that they head for food rather than spin, drift, or win by luck.
- Compare against simple baselines in the same environment: a random-weight network, and a hand-coded "move toward food" controller.

## What is already known (2026-11-07)
- **Headroom in the current env** (10 agents, 200 food, 1000 steps): the ceiling is 20 food per agent. A hand-coded controller that thrusts toward the strongest food sensor (`action = [cos a, sin a, 0]`, `a` = that sector's center angle) scores **19.8** and needs only about 8 input and 16 output weights. Actions are `[forward, lateral, turn]` in the body frame. Turn-only policies orbit food and score about 11–14.
- **Results with `evolve3_zero.py`** (population 40, arenas of 10, 3 episodes per generation, rank-based truncation, top 8 survive, 3 elites):
  - zero init, connection-scale 1: clone-team score 13.5 at generation 19, still rising when stopped
  - zero init, connection-scale 5: 12.7 at generation 9
  - dense random init: stuck at about 11–12 for 35 generations
- With 30 agents on 200 food the average is capped at 6.7. Keep the food-per-agent ratio sensible.

## Known problems in the existing code
1. `evolution3.offspring_count` gives 0 offspring below fitness 0.5, so a zero-init population stalls or dies out. The notebook's steady-state loop (pool of 1,000, sample 10 per generation, children inserted with the parent's unevaluated fitness) barely moves from zero.
2. Dense random init (about 18.8k weights) exceeds `add_remove_connection`'s caps (20% per matrix: 409 Win, 3,276 Wh, 76 Wout), so structural mutation can only remove weights.
3. Low signal gain in Brain3 (`add_sens = 0.01`, `leak = 0.1`): the hidden state is about 0.1× the input drive, so outputs stay near 0 unless weights get large. With a mutation scale of 0.05 this is slow. Likely the main brake.
4. `add_remove_connection` adds at the same per-entry rate it removes, so sparse genomes bloat (0 → 2,600 connections in 100 generations) with no fitness gain.
5. `GenomePE.clone()` builds a random genome and then overwrites it. A `hidden_max` ≤ 0 after mutation silently kills a brain.
6. Every brain runs separately in a Python loop. Batching the forward passes would speed evaluation several-fold, so more generations fit.

## Deliverables
- Working code in `src/`, runnable as `python src/<script>.py`, plus a Colab cell that runs it.
- Best genome checkpoint, fitness logs, a fitness-over-generations plot across seeds, and a rendered video of the best agents.
- A short `RESULTS.md`: what you changed and why, final fitness (mean ± sd over ≥ 20 episodes) against the baselines, connection counts, what mattered, what didn't, and where it still plateaus.
- Commit to a new branch (e.g. `claude/evolve-behaving-agents`) and open a PR. Do not push to `main`. Never commit tokens or credentials.
