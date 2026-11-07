"""Speed: original structure (environment3 + per-agent brain loop, as in
run_episode3) vs the batched numpy simulator, for one generation's worth of
episodes (12 arenas x 10 agents x 1000 steps, sparse ~1k-connection genomes).
torch is not available here, so the per-agent brain is the numpy Brain3 port
called one agent at a time (torch per-call overhead would be larger)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pefe_np as P
from agent3 import Agent
from environment3 import ForagingEnv

rng = np.random.default_rng(0)
gs = [P.Genome.zeros() for _ in range(40)]
for g in gs:
    for _ in range(30):
        g.add_remove_connection(rng, 0.002, 1.0)
E = 12
gi = np.arange(E * 10).reshape(E, 10) % 40
cfg = P.BrainConfig()

t = time.perf_counter()
for e in range(E):
    agents = [Agent() for _ in range(10)]
    brains = [P.BatchBrain([gs[gi[e, a]]], [0], cfg, rng) for a in range(10)]
    env = ForagingEnv(agents, n_food=200, max_steps=1000)
    obs, _ = env.reset(seed=e)
    for _ in range(1000):
        act = np.array([brains[a](obs[a][None].astype(np.float64))[0] for a in range(10)])
        obs, r, term, trunc, _ = env.step(act)
        if term or trunc:
            break
t_orig = time.perf_counter() - t

t = time.perf_counter()
P.run_batch(gs, gi, cfg, P.EnvConfig(), list(range(E)), rng)
t_new = time.perf_counter() - t
print(f"original structure: {t_orig:.1f}s   batched: {t_new:.1f}s   speedup {t_orig / t_new:.1f}x")
