# Reference: evolve.py defaults (= evolve3_zero.py), zero init, 150 generations

Standard eval (clone team of 10, 20 held-out episodes):

| seed | score | blind | connections (best genome) |
|---|---|---|---|
| 0 | 12.00 ± 0.90 | 9.76 ± 2.07 | 2630 |
| 1 | 12.26 ± 0.90 | 1.80 ± 0.99 | 519 |
| 2 | 12.87 ± 0.81 | 4.63 ± 0.85 | 1308 |

- The validation clone-team score is flat at about 12.4 from generation 20 to 150.
- Population mean connections grow linearly to about 3,700 (bloat, problem 4).
- Seed 0's best genome scores almost as well blind as sighted: it mostly drifts.

Hand-wired Brain3 sanity check (`src/handwired.py`, not evolved):
24 weights reach 17.1–17.8 at any gain once weights are large enough
(w_in = c_out = 10). Brain3 has **no bias**, so with no food in view the hidden
state decays to 0 and the agent stops: the network has no tonic exploration
drive. Beating about 17.7 (hand-coded-stop) toward 19.7 needs exploration
through recurrence or the agent sensors.
