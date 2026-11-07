"""Standard evaluation: every experiment reports through this script.

Protocol (fixed so numbers are comparable across experiments):
  * clone team: 10 copies of one controller in one arena
  * 200 food, 1000 steps, world 1000x1000 (environment3 defaults used in the brief)
  * 20 episodes on held-out seeds 1_000_000 .. 1_000_019 (never used in training)
  * score = mean food collected per agent in an episode; we report the mean and
    sd of that over the 20 episodes
  * "blind" = same controller with its 8 food sensors zeroed.  A forager must
    drop sharply when blind; a drifter / spinner scores the same either way.

Usage (from the repo root or src/):
    python src/evaluate.py --baselines
    python src/evaluate.py --genome runs/x/best.npz [--json out.json]
"""

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pefe_np as P  # noqa: E402

EVAL_SEEDS = [1_000_000 + i for i in range(20)]
NOISE_SEED = 12345
BASELINES = ["zero", "forward", "random_dense", "random_sparse",
             "handcoded_stop", "handcoded"]


def _stats(per_episode):
    x = np.asarray(per_episode, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "per_episode": [round(v, 3) for v in x.tolist()]}


def eval_genome(genome, brain_cfg, env_cfg=None, seeds=EVAL_SEEDS, blind=False):
    env_cfg = env_cfg or P.EnvConfig()
    rng = np.random.default_rng(NOISE_SEED)
    gi = np.zeros((len(seeds), env_cfg.n_agents), int)
    food = P.run_batch([genome], gi, brain_cfg, env_cfg, seeds, rng, blind=blind)
    return food.mean(1)  # per-episode mean food per agent


def eval_controller(fn, env_cfg=None, seeds=EVAL_SEEDS, blind=False):
    env_cfg = env_cfg or P.EnvConfig()
    rng = np.random.default_rng(NOISE_SEED)
    food = P.run_batch(None, None, None, env_cfg, seeds, rng, controller=fn, blind=blind)
    return food.mean(1)


def eval_random_genomes(make, brain_cfg, env_cfg=None, seeds=EVAL_SEEDS, blind=False):
    """One fresh random genome per episode (clone team of it)."""
    env_cfg = env_cfg or P.EnvConfig()
    rng = np.random.default_rng(777)
    genomes = [make(rng) for _ in seeds]
    gi = np.repeat(np.arange(len(seeds))[:, None], env_cfg.n_agents, 1)
    food = P.run_batch(genomes, gi, brain_cfg, env_cfg, seeds,
                       np.random.default_rng(NOISE_SEED), blind=blind)
    return food.mean(1), float(np.mean([g.n_connections() for g in genomes]))


def random_sparse(rng, n=50, scale=1.0):
    g = P.Genome.zeros()
    flat = rng.choice(P.NUM_PARAMETERS_FULL, n, replace=False)
    sizes = np.cumsum([0, P.N_WIN, P.N_WH, P.N_WOUT])
    for (_, W, _), lo, hi in zip(g.matrices(), sizes[:-1], sizes[1:]):
        sel = flat[(flat >= lo) & (flat < hi)] - lo
        W.flat[sel] = rng.standard_normal(len(sel)) * scale
    return g


def run_baseline(name, env_cfg=None):
    brain_cfg = P.BrainConfig()
    out = {"name": name, "protocol": "clone10_f200_s1000_20ep"}
    if name == "handcoded":
        f = P.handcoded_actions
    elif name == "handcoded_stop":
        f = P.handcoded_stop_actions
    elif name == "forward":
        f = lambda o: np.broadcast_to(np.array([1.0, 0, 0]), o.shape[:-1] + (3,)).copy()  # noqa
    elif name == "zero":
        f = lambda o: np.zeros(o.shape[:-1] + (3,))  # noqa
    elif name in ("random_dense", "random_sparse"):
        make = P.Genome.random_dense if name == "random_dense" else random_sparse
        s, conns = eval_random_genomes(make, brain_cfg, env_cfg)
        b, _ = eval_random_genomes(make, brain_cfg, env_cfg, blind=True)
        out.update(score=_stats(s), blind=_stats(b), connections=conns)
        return out
    else:
        raise ValueError(name)
    out.update(score=_stats(eval_controller(f, env_cfg)),
               blind=_stats(eval_controller(f, env_cfg, blind=True)),
               connections=None)
    return out


def run_genome(path, env_cfg=None):
    g, cfg, meta = P.load_genome(path)
    s = eval_genome(g, cfg, env_cfg)
    b = eval_genome(g, cfg, env_cfg, blind=True)
    return {"name": path, "protocol": "clone10_f200_s1000_20ep", "score": _stats(s),
            "blind": _stats(b), "connections": g.n_connections(),
            "connection_counts": g.connection_counts(), "brain_config": cfg.__dict__,
            "meta": meta}


def fmt(r):
    c = r["connections"]
    c = "-" if c is None else f"{c:.0f}"
    return (f"{os.path.basename(str(r['name'])):<28} {r['score']['mean']:6.2f} ± {r['score']['sd']:5.2f}"
            f"   blind {r['blind']['mean']:6.2f} ± {r['blind']['sd']:5.2f}   conns {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args()
    results = []
    if a.baselines:
        for n in BASELINES:
            results.append(run_baseline(n))
            print(fmt(results[-1]), flush=True)
    for path in a.genome:
        results.append(run_genome(path))
        print(fmt(results[-1]), flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
