"""Sanity check (NOT an evolved result): hand-wired Brain3 genomes that
approximate the hand-coded controller, to check what Brain3 can represent
at a given gain.  Food sensor k -> hidden k (weight w_in), hidden k ->
forward/lateral (c*cos a_k, c*sin a_k).  Optional self-loop on hidden k
gives persistence (keeps moving after food leaves view).  24 or 32 weights."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np
import pefe_np as P
from evaluate import eval_genome


def handwired(w_in, c_out, self_loop=0.0):
    g = P.Genome.zeros()
    for k in range(8):
        g.Win[k, k] = w_in
        g.Wout[0, k] = c_out * np.cos(P.SECTOR_CENTERS[k])
        g.Wout[1, k] = c_out * np.sin(P.SECTOR_CENTERS[k])
        if self_loop:
            g.Wh[k, k] = self_loop
    return g


if __name__ == "__main__":
    for add_sens in (0.01, 0.1, 1.0):
        cfg = P.BrainConfig(add_sens=add_sens)
        for w, c, sl in ((1, 1, 0), (10, 10, 0), (100, 10, 0), (10, 10, 0.18)):
            g = handwired(w, c, sl)
            s = eval_genome(g, cfg)
            b = eval_genome(g, cfg, blind=True)
            print(f"add_sens {add_sens:5}  w_in {w:4} c_out {c:3} self {sl:4}  conns {g.n_connections():3d}"
                  f"  score {s.mean():5.2f} ± {s.std(ddof=1):4.2f}  blind {b.mean():5.2f}", flush=True)
