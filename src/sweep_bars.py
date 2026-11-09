"""Measure how much the bars make navigation harder (v2 world).

Runs the hand-coded "thrust toward the strongest food sensor" baseline (an
evaluation reference only, never used for training) on the 20 held-out
evaluation episodes for several (number of bars, bar length) settings, and
records the bar statistics.  0 bars is exactly the v1 world.

    python src/sweep_bars.py --json results/v2/bar_sweep.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pefe_np as P  # noqa: E402
import pefe_blocks as BL  # noqa: E402
from evaluate import EVAL_SEEDS, NOISE_SEED  # noqa: E402

SETTINGS = [(0, 300), (10, 200), (20, 200), (20, 300), (30, 200), (30, 300),
            (40, 200), (40, 250), (40, 300), (50, 250), (60, 200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = []
    for nb, length in SETTINGS:
        cfg = BL.BlockEnvConfig(n_bars=nb, bar_length=length)
        envs = []
        food = P.run_batch(None, None, None, cfg, EVAL_SEEDS, np.random.default_rng(NOISE_SEED),
                           controller=P.handcoded_actions, env_out=envs).mean(1)
        bars = BL.summarize_bar_stats(envs[0].bar_stats(), cfg)
        r = {"n_bars": nb, "bar_length": length, "area_frac":
             nb * (length * 2 * cfg.bar_radius + np.pi * cfg.bar_radius ** 2) / cfg.world_size ** 2,
             "handcoded_mean": float(food.mean()), "handcoded_sd": float(food.std(ddof=1)),
             "bars": bars}
        out.append(r)
        print(f"{nb:3d} bars x {length:4.0f}  area {100 * r['area_frac']:4.1f}%  handcoded "
              f"{food.mean():6.2f} ± {food.std(ddof=1):4.2f}   bar disp {bars['net_disp_mean']:5.1f}"
              f"  >=2 pushers {bars['multi_push_bar_steps']:6.0f} bar-steps/ep", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
