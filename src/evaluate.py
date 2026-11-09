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
    python src/evaluate.py --baselines --world v2      # v2 world with bars

v2 ("blocks", src/pefe_blocks.py): the same protocol in the world with
movable bars.  A genome with 24 inputs is evaluated in v2 automatically, with
the bar settings saved in its meta (else the BlockEnvConfig defaults).  v2
results also report bar statistics (how far bars moved, how often agents
pushed, simultaneous pushers) and a "bar-blind" score with the 8 bar sensors
zeroed.
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

EVAL_SEEDS = [1_000_000 + i for i in range(20)]
NOISE_SEED = 12345
BASELINES = ["zero", "forward", "random_dense", "random_sparse",
             "handcoded_stop", "handcoded"]


def _stats(per_episode):
    x = np.asarray(per_episode, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "per_episode": [round(v, 3) for v in x.tolist()]}


def _bar_summary(env_out, env_cfg):
    if env_out and hasattr(env_out[-1], "bar_stats"):
        return BL.summarize_bar_stats(env_out[-1].bar_stats(), env_cfg)
    return None


def eval_genome(genome, brain_cfg, env_cfg=None, seeds=EVAL_SEEDS, blind=False,
                env_out=None):
    env_cfg = env_cfg or P.EnvConfig()
    rng = np.random.default_rng(NOISE_SEED)
    gi = np.zeros((len(seeds), env_cfg.n_agents), int)
    food = P.run_batch([genome], gi, brain_cfg, env_cfg, seeds, rng, blind=blind,
                       env_out=env_out)
    return food.mean(1)  # per-episode mean food per agent


def eval_controller(fn, env_cfg=None, seeds=EVAL_SEEDS, blind=False, env_out=None):
    env_cfg = env_cfg or P.EnvConfig()
    rng = np.random.default_rng(NOISE_SEED)
    food = P.run_batch(None, None, None, env_cfg, seeds, rng, controller=fn, blind=blind,
                       env_out=env_out)
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


def world_config(world, genome=None, meta=None):
    """Env config for evaluation: 'v1', 'v2' or 'auto' (from the genome)."""
    if world == "auto":
        world = "v2" if genome is not None and genome.n_inputs == BL.N_INPUTS_V2 else "v1"
    if world == "v1":
        return P.EnvConfig()
    env = dict((meta or {}).get("env", {}))
    # evaluation protocol is fixed: clone team of 10, 200 food, 1000 steps
    env.update(n_agents=10, n_food=200, max_steps=1000)
    return BL.BlockEnvConfig(**env)


def random_sparse(rng, n=50, scale=1.0, n_inputs=P.INPUT_SIZE):
    g = P.Genome.zeros(n_inputs)
    n_win = P.HIDDEN_SIZE * n_inputs
    flat = rng.choice(n_win + P.N_WH + P.N_WOUT, n, replace=False)
    sizes = np.cumsum([0, n_win, P.N_WH, P.N_WOUT])
    for (_, W, _), lo, hi in zip(g.matrices(), sizes[:-1], sizes[1:]):
        sel = flat[(flat >= lo) & (flat < hi)] - lo
        W.flat[sel] = rng.standard_normal(len(sel)) * scale
    return g


def run_baseline(name, env_cfg=None):
    brain_cfg = P.BrainConfig()
    n_in = (env_cfg or P.EnvConfig()).n_inputs
    out = {"name": name, "protocol": "clone10_f200_s1000_20ep"}
    if n_in != P.INPUT_SIZE:
        out["world"] = "v2"
        out["env"] = BL.asdict(env_cfg)
    if name == "handcoded":
        f = P.handcoded_actions
    elif name == "handcoded_stop":
        f = P.handcoded_stop_actions
    elif name == "forward":
        f = lambda o: np.broadcast_to(np.array([1.0, 0, 0]), o.shape[:-1] + (3,)).copy()  # noqa
    elif name == "zero":
        f = lambda o: np.zeros(o.shape[:-1] + (3,))  # noqa
    elif name in ("random_dense", "random_sparse"):
        if name == "random_dense":
            make = lambda r: P.Genome.random_dense(r, n_inputs=n_in)  # noqa
        else:
            make = lambda r: random_sparse(r, n_inputs=n_in)  # noqa
        s, conns = eval_random_genomes(make, brain_cfg, env_cfg)
        b, _ = eval_random_genomes(make, brain_cfg, env_cfg, blind=True)
        out.update(score=_stats(s), blind=_stats(b), connections=conns)
        return out
    else:
        raise ValueError(name)
    envs = []
    out.update(score=_stats(eval_controller(f, env_cfg, env_out=envs)),
               blind=_stats(eval_controller(f, env_cfg, blind=True)),
               connections=None)
    bars = _bar_summary(envs, env_cfg)
    if bars is not None:
        out["bars"] = bars
    return out


def run_genome(path, env_cfg=None, world="auto"):
    g, cfg, meta = P.load_genome(path)
    if env_cfg is None:
        env_cfg = world_config(world, g, meta)
    if g.n_inputs < env_cfg.n_inputs:
        # a v1 genome in the v2 world: zero weights from the extra (bar) sensors
        Win = np.zeros((P.HIDDEN_SIZE, env_cfg.n_inputs), np.float32)
        Win[:, :g.n_inputs] = g.Win
        g = P.Genome(Win, g.Wh, g.Wout, g.hyper)
    envs = []
    s = eval_genome(g, cfg, env_cfg, env_out=envs)
    b = eval_genome(g, cfg, env_cfg, blind=True)
    out = {"name": path, "protocol": "clone10_f200_s1000_20ep", "score": _stats(s),
           "blind": _stats(b), "connections": g.n_connections(),
           "connection_counts": g.connection_counts(), "brain_config": cfg.__dict__,
           "meta": meta}
    bars = _bar_summary(envs, env_cfg)
    if bars is not None:  # v2: bar statistics and the bar-blind check
        out["world"] = "v2"
        out["env"] = BL.asdict(env_cfg)
        out["bars"] = bars
        rng = np.random.default_rng(NOISE_SEED)
        gi = np.zeros((len(EVAL_SEEDS), env_cfg.n_agents), int)
        food = P.run_batch([g], gi, cfg, env_cfg, EVAL_SEEDS, rng, zero_inputs=slice(16, 24))
        out["bar_blind"] = _stats(food.mean(1))
    return out


def fmt(r):
    c = r["connections"]
    c = "-" if c is None else f"{c:.0f}"
    s = (f"{os.path.basename(str(r['name'])):<28} {r['score']['mean']:6.2f} ± {r['score']['sd']:5.2f}"
         f"   blind {r['blind']['mean']:6.2f} ± {r['blind']['sd']:5.2f}   conns {c}")
    if "bar_blind" in r:
        s += f"   bar-blind {r['bar_blind']['mean']:6.2f}"
    if "bars" in r:
        b = r["bars"]
        s += (f"\n{'':28} bars: net disp {b['net_disp_mean']:.1f} (max {b['net_disp_max']:.0f}),"
              f" moved>10 {b['bars_moved_10']:.1f}/ep, push {100 * b['push_frac']:.1f}% of agent-steps,"
              f" >=2 pushers {b['multi_push_bar_steps']:.0f} bar-steps/ep"
              f" ({100 * b['episodes_with_multi']:.0f}% of eps, max {b['max_pushers']})")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--world", choices=["auto", "v1", "v2"], default="auto",
                    help="auto: v2 for 24-input genomes, else v1 (baselines: v1); "
                         "--world v2 with a v1 genome runs it in v2 without bar sensors")
    a = ap.parse_args()
    results = []
    if a.baselines:
        env_cfg = world_config("v1" if a.world == "auto" else a.world)
        for n in BASELINES:
            results.append(run_baseline(n, env_cfg))
            print(fmt(results[-1]), flush=True)
    for path in a.genome:
        results.append(run_genome(path, world=a.world))
        print(fmt(results[-1]), flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
