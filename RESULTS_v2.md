# PEfE v2: foraging among movable bars. Results

The v2 world adds **40 heavy bars of length 300** to the v1 foraging world.
Agents cannot pass through a bar, but they can push it. The setting is chosen
so navigation is clearly harder: the hand-coded "thrust toward food"
controller drops from **19.70 to 14.12** food per agent. Evolution from
all-zero genomes, with the v1 loop and defaults and 24 sensor inputs,
**beats that controller on all 3 seeds: 15.18, 16.32 and 15.57**. It also
beats the v1-evolved agents moved into this world (13.1–15.2). All behaviour
is evolved. The hand-coded controllers are evaluation references only and are
never used in training.

## The v2 world (`src/pefe_blocks.py`, `--world v2`)
- **Bars**: capsules (a segment plus radius 5, so 10 thick) at random positions
  and angles. Each bar is a rigid body with position, angle, linear and angular
  velocity. Mass is 50 (an agent has mass 1), and the moment of inertia is
  that of a rod, M·L²/12. Both linear and angular velocity are multiplied by
  0.9 every step (strong damping), so a bar stops within about 50 steps once
  nobody pushes it.
- **Contacts**: when an agent overlaps a bar, an inelastic impulse cancels the
  approaching part of the relative velocity. The impulse uses the full
  rigid-body effective mass, so an off-centre push also produces torque. The
  bar receives the opposite impulse, and all pushers' impulses are summed. The
  agent is then moved fully out of the bar. If an agent is squeezed between a
  bar and a wall, the bar is moved instead.
- **No crossing**: an agent moves at most 5 per step, which is less than the
  contact distance of 10 (agent radius plus bar radius), so it can never jump
  past a bar's axis.
- **Walls**: bars are clamped inside the world, and the velocity component
  into the wall is zeroed.
- **Bar–bar overlap (the simple choice)**: bars spawn with at least 12 units
  between their surfaces. During an episode, each overlapping pair gets one
  inelastic impulse plus half-and-half positional separation per step, with no
  iteration. Bars therefore never pass through each other, but a chain of bars
  can overlap slightly for a step.
- **Unchanged from v1**: bars do not block sensing, and agents still pass
  through each other. Food under a bar's axis is still reachable: the eat
  radius is 13 and the closest an agent centre can get is 10.
- **Measured push strength** (`tests/test_blocks.py` setup, 200 steps of
  full-power pushing):

| pushers | bar speed (per step) | displacement | rotation |
|---|---|---|---|
| 1, centre | 0.20 (4% of agent max speed) | 34 | 0° |
| 2, centre | 0.39 | 67 | 0° |
| 3, centre | 0.58 | 100 | 0° |
| 1, near an end (80 of 150 from centre) | – | 15 | 11° |

- **Sensors**: 24 inputs = [8 food, 8 agents, 8 bars]. Bar sector k holds
  1 − d/150 for the nearest bar surface in that sector. It is computed from
  points every 10 units along each bar's axis plus the exact nearest point
  (d = axis distance − bar radius).
- **Random streams**: food and agent positions use exactly the v1 draws
  (`default_rng(seed)`). Bars use a separate stream,
  `default_rng([seed, 0xB10C5])`. Agents that spawn inside a bar are nudged
  out along the normal. A test checks that v2 with 0 bars reproduces v1
  exactly (19.70).

## Choosing the number and length of bars
The hand-coded controller (thrust toward the strongest food sensor) on the 20
held-out episodes (`python src/sweep_bars.py`, `results/v2/bar_sweep.json`):

| bars × length | area covered | hand-coded score | mean bar displacement |
|---|---|---|---|
| 0 (= v1) | 0% | 19.70 ± 0.66 | – |
| 10 × 200 | 2.1% | 19.66 ± 0.49 | 22.0 |
| 20 × 200 | 4.2% | 19.73 ± 0.44 | 22.3 |
| 20 × 300 | 6.2% | 17.81 ± 1.79 | 37.7 |
| 30 × 200 | 6.2% | 18.91 ± 1.00 | 22.9 |
| 30 × 300 | 9.2% | 16.39 ± 1.95 | 28.8 |
| 40 × 200 | 8.3% | 17.78 ± 1.40 | 21.0 |
| 40 × 250 | 10.3% | 16.16 ± 1.83 | 23.7 |
| **40 × 300 (chosen)** | 12.3% | **14.12 ± 2.11** | 26.4 |
| 50 × 250 | 12.9% | 13.96 ± 2.05 | 21.6 |
| 60 × 200 | 12.5% | 15.49 ± 1.56 | 17.4 |

Contacts are frictionless, so a controller that thrusts at a slant slides
along a bar. Short bars therefore cost almost nothing, and length matters more
than count. 40 × 300 gives a 28% drop and is cheaper to simulate than 50 × 250
for the same effect. 50 bars of length 300 sometimes cannot be placed.

## Evolution
Same loop and defaults as v1 (`src/evolve.py`): zero init, about 3 new
connections per child (40/20/40), connection scale 10, removal 0.02,
population 40, arenas of 10, 3 episodes per generation, top 8 survive,
3 elites. Win is now 128 × 24. The only change is **400 generations instead
of 150**, because fitness was still rising at 150 (see below). One generation
takes about 4–6 s, so a run takes about 33 minutes on one core.

```bash
python src/evolve.py --out runs/v2/s0 --world v2 --seed 0 --generations 400 --checkpoint-every 50
```

## Final numbers (standard evaluation: clone team of 10, 20 held-out episodes)
`blind` = food sensors zeroed. `bar-blind` = bar sensors zeroed.

| controller (v2 world) | score | blind | bar-blind | connections (Win/Wh/Wout) |
|---|---|---|---|---|
| **evolved, seed 0 (400 gen)** | **15.18 ± 1.15** | 6.85 | 10.21 | 165 (70/29/66) |
| **evolved, seed 1 (400 gen)** | **16.32 ± 0.93** | 6.44 | 2.78 | 136 (55/26/55) |
| **evolved, seed 2 (400 gen)** | **15.57 ± 0.98** | 7.68 | 10.91 | 140 (51/27/62) |
| evolved, seeds 0/1/2 at generation 150 (elite) | 12.95 / 12.27 / 15.18 | 4.06 / 8.89 / 6.07 | 10.41 / 8.68 / 12.02 | 139 / 140 / 140 |
| v1-evolved genomes (`results/final`), no bar sensors | 13.14 / 15.15 / 13.76 | 0.55 / 0.68 / 0.43 | – | 146 / 149 / 147 |
| hand-coded, thrust toward food | 14.12 ± 2.11 | 3.23 | – | – |
| hand-coded, same but stops with no food in view | 13.32 ± 1.70 | 0.14 | – | – |
| random dense network | 6.09 ± 2.03 | 5.83 | – | 19,840 |
| always forward | 3.33 ± 0.46 | 3.33 | – | – |
| zero output | 0.14 ± 0.11 | 0.14 | – | – |

For reference, in v1 the same pipeline reaches 18.2–18.7 and the hand-coded
controller 19.70. The bars cost the evolved agents about 3 food per agent and
the hand-coded controller about 5.6.

- **They use the bar sensors.** Zeroing the 8 bar inputs drops the score to
  10.2, 2.8 and 10.9 on the three seeds. The v1 genomes, which have no bar
  sensors, score 1–3 lower.
- **The blind check is higher than in v1 (6.4–7.7, against about 0.6).** The
  agents do forage with their food sensors, since zeroing those halves the
  score. But about 45% of their food comes from a search pattern that does not
  need food sensing. The blind trajectories (`results/v2/best_s1_blind_trails.png`)
  show agents running loops along and around bars. This is not the drifting
  failure the blind check was designed to catch in v1: the zero-genome,
  forward and random baselines score 0.1–6. It is an evolved exploration
  strategy that uses the bars as landmarks.
- **More generations helped.** The held-out scores of the generation-150
  elites are 12.95, 12.27 and 15.18, only around the hand-coded controller's
  level. At 400 generations they are 15.18, 16.32 and 15.57. The population
  mean per 50-generation block was still rising slowly on seeds 0 and 1 at
  generation 400 (seed 0: 13.2 → 13.6, seed 1: 14.6 → 15.1). Seed 2 levelled
  off at about 14 from generation 100. Curves: `results/v2/curves.png`.
- Connection counts stay at about 140–165, as in v1. Removal still controls
  bloat.

## Do agents move bars, and do they push together?
Bar statistics over the 20 held-out episodes (`bars` in the eval JSONs).
"Pushing" means an agent touches a bar with a non-zero contact impulse.

| controller | push (% of agent-steps) | bars moved > 10 per episode (of 40) | mean / max net displacement | mean rotation | bar-steps with ≥ 2 pushers per episode | most simultaneous pushers | share of bar movement made while ≥ 2 pushed |
|---|---|---|---|---|---|---|---|
| evolved s0 | 41% | 17.2 | 11.5 / 108 | 4.6° | 231 | 3 | 5.7% |
| evolved s1 | 50% | 17.8 | 12.9 / 162 | 5.1° | 364 | 4 | 10.6% |
| evolved s2 | 56% | 21.3 | 16.2 / 187 | 6.0° | 404 | 4 | 8.5% |
| v1 genomes s0/s1/s2 | 60 / 55 / 40% | 19.5 / 21.6 / 16.8 | 17.7 / 18.4 / 13.1 | 4–7° | 333 / 439 / 132 | 3 | 4–9% |
| hand-coded | 82% | 24.8 | 26.4 / 178 | 9.4° | 1009 | 4 | 13.7% |
| always forward | 14% | 5.1 | 4.3 / 151 | 1.7° | 113 | 3 | 9.0% |

- **How often and how far.** Evolved agents are in pushing contact with a bar
  for 41–56% of their steps. In a typical episode 17–21 of the 40 bars move
  more than 10 units. A bar's centre ends on average 11–16 units from where it
  started (path length 20–30), and the furthest moves are 108–187 units. Bars
  rotate by about 5° on average.
- **Several agents pushing together: yes, every episode.** There are 230–400
  bar-steps per episode with two or more simultaneous pushers, with up to
  3–4 agents on one bar at once. Those steps account for 6–11% of all bar
  movement.
- **This is not evidence of cooperation.** The hand-coded controller has no
  notion of bars or teammates, yet it has 2–4 times more joint pushing (1009
  bar-steps per episode). Joint pushes happen because clone teammates head for
  the same food, often from the same side of a bar. Evolution *reduced*
  pushing compared with the hand-coded agent (41–56% against 82% of steps):
  being stuck against a bar costs time. Bars move as a side effect of
  navigation, not as a goal. The fitness (food only) gives no reason to move
  bars deliberately.

## Video and plots
- `results/v2/best_s1.mp4`: the seed-1 team on the first held-out seed. Bars
  are brown, their start positions are dashed outlines, and a bar turns orange
  while two or more agents push it.
- `results/v2/best_s1_trails.png` and `results/v2/best_s1_blind_trails.png`:
  trajectories with the start positions of the bars (grey) and their end
  positions (brown).
- `results/v2/curves.png`: fitness, validation and connection curves for the
  3 seeds.
- `results/v2/s{0,1,2}/`: config, `log.csv` (with per-generation bar columns
  `bar_disp`, `bars_moved`, `push_frac`, `multi_steps`), `best.npz`,
  `gen150_elite.npz` and `eval.json`.
- `results/v2/baselines_v2.json`, `bar_sweep.json`, `v1_genomes_in_v2.json`.

## Reproduce
```bash
python src/evaluate.py --baselines --world v2 --json results/v2/baselines_v2.json
```

```bash
python src/evaluate.py --world v2 --genome results/final/s0/best.npz results/final/s1/best.npz results/final/s2/best.npz
```

```bash
python src/evaluate.py --genome results/v2/s1/best.npz
```

```bash
python src/render.py --genome results/v2/s1/best.npz --out results/v2/best_s1.mp4
```

Colab: `PEfE_v2_blocks_colab.ipynb` (`BRANCH = 'v2-blocks'`).

## Compatibility with v1
v1 is the default for every script (`--world v1`). Its tests pass unchanged:
`test_equivalence.py`, and `test_defaults.py`, which still reproduces the
golden 40-generation log row for row. The only changes to v1 code are
refactors: the genome input size is now read from the genome, `BatchEnv.step`
is split into move and eat, and `run_batch` gained optional `env_out` and
`zero_inputs` arguments. `tests/test_blocks.py` checks that v2 with 0 bars
gives exactly the v1 result.

## Possible next steps
- Fitness gives no reason to move bars on purpose. A task that needs it, such
  as food behind barriers or a reward for opening a passage, would show
  whether deliberate and cooperative pushing can evolve.
- Contact friction (no sliding along bars) or bars that block sensing would
  make navigation harder still.
