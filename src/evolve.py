"""Configurable evolution driver on the batched numpy simulator.

Defaults are the final "B apc" setup: generational, population 40, arenas of
10, 3 episodes/gen, rank-weighted truncation, top 8 survive, 3 elites, Brain3
default gains, zero-init genomes, 150 generations, and the B structural
mutation: about 3 new connections per child (split 40% Win / 20% Wh / 40% Wout)
with strong initial weights (connection scale 10) and per-entry removal 0.02.
The full table is in README.md.  To get the older evolve3_zero.py behaviour
(per-entry add = remove = 0.002, scale 1) pass
    --add-per-child none --add-share none --remove-rate none --connection-scale 1
Every knob is a flag, so runs differ only by their command line (saved in
config.json).

Outputs in --out:
  config.json     full argument set
  log.csv         one row per generation
  best.npz        genome with the best *validation* clone-team score
  final_pop.npz   last population
Final numbers must come from evaluate.py (held-out seeds), not from log.csv.

Usage:
    python src/evolve.py --out runs/x --seed 0
    python src/evolve.py --out runs/v2 --seed 0 --world v2   # movable bars

--world v2 runs the "blocks" world of src/pefe_blocks.py (24 sensor inputs:
food, agents, bars).  Everything else in the loop is the same; the bar
geometry / physics flags (--n-bars ...) are used only with --world v2, and
log.csv gets extra bar columns only then.
"""

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pefe_np as P  # noqa: E402
import pefe_blocks as BL  # noqa: E402

_V2 = BL.BlockEnvConfig()  # defaults of the v2 bar flags

VALID_SEED_BASE = 500_000  # validation seeds; evaluate.py uses 1_000_000+


def _float_or_none(x):
    return None if str(x).lower() == "none" else float(x)


def _str_or_none(x):
    return None if str(x).lower() == "none" else x


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--out", required=True)
    a("--seed", type=int, default=0)
    a("--generations", type=int, default=150)
    a("--max-evals", type=int, default=0,
      help="stop after this many agent-episodes (0 = no limit); use it to "
           "compare schemes at equal compute")
    # population / selection
    a("--population", type=int, default=40)
    a("--arena-size", type=int, default=10)
    a("--episodes", type=int, default=3, help="episodes per generation")
    a("--survivors", type=int, default=8)
    a("--elites", type=int, default=3)
    a("--eval-mode", choices=["mixed", "clone"], default="mixed",
      help="mixed: shuffled arenas of different genomes; clone: each genome "
           "is evaluated as a team of --arena-size copies of itself")
    # init
    a("--init", default="zeros", help="zeros | random | sparse:N")
    # environment
    a("--food", type=int, default=200)
    a("--steps", type=int, default=1000)
    a("--world", choices=["v1", "v2"], default="v1",
      help="v1: the original foraging world; v2: plus movable bars (pefe_blocks)")
    a("--n-bars", type=int, default=_V2.n_bars, help="(v2) number of bars")
    a("--bar-length", type=float, default=_V2.bar_length, help="(v2) bar length")
    a("--bar-radius", type=float, default=_V2.bar_radius, help="(v2) half thickness")
    a("--bar-mass", type=float, default=_V2.bar_mass, help="(v2) bar mass (agent = 1)")
    a("--bar-damping", type=float, default=_V2.bar_damping,
      help="(v2) per-step velocity multiplier (linear and angular)")
    # brain gains (Brain3 constants)
    a("--hidden-max", type=float, default=10000.0)
    a("--add-hidden", type=float, default=0.5)
    a("--add-sens", type=float, default=0.01)
    a("--leak-hidden", type=float, default=0.1)
    a("--max-epsilon", type=float, default=1.0)
    a("--no-noise", action="store_true", help="disable Brain3 output noise")
    # mutation
    a("--mutation-rate", type=float, default=0.1)
    a("--mutation-scale", type=float, default=0.05)
    a("--no-hyper-mutation", action="store_true")
    a("--connection-rate", type=float, default=0.002, help="per-entry add prob.")
    a("--remove-rate", type=_float_or_none, default=0.02,
      help="per-entry remove prob.; 'none' = same as --connection-rate, "
           "as in genome_pe3")
    a("--connection-scale", type=float, default=10.0,
      help="sd of the weight given to a newly added connection")
    a("--no-caps", action="store_true", help="drop the 20%% per-matrix caps")
    a("--add-per-child", type=_float_or_none, default=3.0,
      help="expected number of added connections per child (replaces the "
           "per-entry --connection-rate for additions); 'none' = use "
           "--connection-rate per entry")
    a("--add-share", type=_str_or_none, default="0.4,0.2,0.4",
      help="with --add-per-child: comma-separated Win,Wh,Wout split of the "
           "additions; 'none' = proportional to matrix size")
    a("--add-path-prob", type=float, default=0.0,
      help="per-child probability of adding one input->hidden->output path "
           "(a Win and a Wout entry through the same random hidden unit)")
    a("--connection-cost", type=float, default=0.0,
      help="selection fitness = food - cost * n_connections")
    # validation / checkpoints
    a("--check-every", type=int, default=5)
    a("--check-episodes", type=int, default=5)
    a("--checkpoint-every", type=int, default=0)
    return p


def brain_config(args):
    return P.BrainConfig(hidden_max=args.hidden_max, add_hidden=args.add_hidden,
                         add_sens=args.add_sens, leak_hidden=args.leak_hidden,
                         max_epsilon=args.max_epsilon, noise=not args.no_noise)


def env_config(args):
    if getattr(args, "world", "v1") == "v2":
        return BL.BlockEnvConfig(n_agents=args.arena_size, n_food=args.food,
                                 max_steps=args.steps, n_bars=args.n_bars,
                                 bar_length=args.bar_length, bar_radius=args.bar_radius,
                                 bar_mass=args.bar_mass, bar_damping=args.bar_damping,
                                 bar_ang_damping=args.bar_damping)
    return P.EnvConfig(n_agents=args.arena_size, n_food=args.food, max_steps=args.steps)


def init_genome(args, rng, n_inputs=P.INPUT_SIZE):
    if args.init == "zeros":
        return P.Genome.zeros(n_inputs)
    if args.init == "random":
        return P.Genome.random_dense(rng, n_inputs=n_inputs)
    if args.init.startswith("sparse:"):
        from evaluate import random_sparse
        return random_sparse(rng, int(args.init.split(":")[1]), args.connection_scale,
                             n_inputs=n_inputs)
    raise ValueError(args.init)


def evaluate_population(genomes, args, rng, bcfg, ecfg, env_out=None):
    """Return (mean food per genome, number of agent-episodes used).
    env_out: optional list that receives the finished environment (v2 stats)."""
    n, A = len(genomes), args.arena_size
    if args.eval_mode == "mixed":
        layout, seeds = [], []
        for _ in range(args.episodes):
            order = rng.permutation(n)
            for s in range(0, n, A):
                layout.append(order[s:s + A])
                seeds.append(int(rng.integers(2**31)))
        layout = np.array(layout)
        food = P.run_batch(genomes, layout, bcfg, ecfg, seeds, rng, env_out=env_out)
        totals = np.bincount(layout.ravel(), weights=food.ravel(), minlength=n)
        counts = np.bincount(layout.ravel(), minlength=n)
        return totals / counts, layout.size
    # clone teams: each genome in its own arena, `episodes` episodes
    layout = np.repeat(np.repeat(np.arange(n), args.episodes)[:, None], A, 1)
    seeds = [int(s) for s in rng.integers(2**31, size=len(layout))]
    food = P.run_batch(genomes, layout, bcfg, ecfg, seeds, rng, env_out=env_out)
    per_arena = food.mean(1)
    return np.bincount(layout[:, 0], weights=per_arena, minlength=n) / args.episodes, layout.size


def validate(genome, args, bcfg, ecfg, gen):
    """Clone-team score on validation seeds (not the held-out test seeds)."""
    seeds = [VALID_SEED_BASE + 1000 * gen + i for i in range(args.check_episodes)]
    gi = np.zeros((len(seeds), args.arena_size), int)
    food = P.run_batch([genome], gi, bcfg, ecfg, seeds, np.random.default_rng(gen))
    return float(food.mean())


def make_child(parent, args, rng):
    child = parent.clone()
    child.mutate(rng, args.mutation_rate, args.mutation_scale,
                 mutate_hyper=not args.no_hyper_mutation)
    child.add_remove_connection(rng, args.connection_rate, args.connection_scale,
                                remove_rate=args.remove_rate, caps=not args.no_caps,
                                add_per_child=args.add_per_child,
                                add_share=(None if args.add_share is None else
                                           [float(x) for x in args.add_share.split(",")]))
    if args.add_path_prob > 0 and rng.random() < args.add_path_prob:
        child.add_path(rng, args.connection_scale)
    return child


def next_generation(genomes, sel_fitness, args, rng):
    order = np.lexsort((rng.random(len(genomes)), -sel_fitness))
    survivors = [genomes[i] for i in order[:args.survivors]]
    new = [g.clone() for g in survivors[:args.elites]]
    w = np.arange(len(survivors), 0, -1, dtype=float)
    w /= w.sum()
    while len(new) < args.population:
        new.append(make_child(survivors[rng.choice(len(survivors), p=w)], args, rng))
    return new


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.eval_mode == "mixed" and args.population % args.arena_size:
        raise SystemExit("--population must be a multiple of --arena-size")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    rng = np.random.default_rng(args.seed)
    bcfg, ecfg = brain_config(args), env_config(args)
    v2 = args.world == "v2"
    genomes = [init_genome(args, rng, ecfg.n_inputs) for _ in range(args.population)]
    meta_world = {"world": "v2", "env": BL.asdict(ecfg)} if v2 else {}

    log_path = os.path.join(args.out, "log.csv")
    cols = ["generation", "evals", "best", "mean", "median", "best_connections",
            "mean_connections", "valid", "best_valid", "seconds"]
    if v2:  # population-wide bar statistics of this generation's episodes
        cols += ["bar_disp", "bars_moved", "push_frac", "multi_steps"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    evals, best_valid = 0, -1.0
    t_start = time.perf_counter()
    for gen in range(args.generations):
        t0 = time.perf_counter()
        envs = [] if v2 else None
        fitness, used = evaluate_population(genomes, args, rng, bcfg, ecfg, envs)
        evals += used
        conns = np.array([g.n_connections() for g in genomes])
        sel = fitness - args.connection_cost * conns
        b = int(np.argmax(sel))

        valid = ""
        last = gen == args.generations - 1 or (args.max_evals and evals >= args.max_evals)
        if (gen + 1) % args.check_every == 0 or last:
            v = validate(genomes[b], args, bcfg, ecfg, gen)
            valid = f"{v:.3f}"
            if v > best_valid:
                best_valid = v
                P.save_genome(os.path.join(args.out, "best.npz"), genomes[b], bcfg,
                              {"generation": gen, "valid": v, "evals": evals,
                               "run": args.out, **meta_world})

        if args.checkpoint_every and (gen + 1) % args.checkpoint_every == 0:
            _save_pop(os.path.join(args.out, f"pop_{gen + 1}.npz"), genomes, fitness)

        dt = time.perf_counter() - t0
        row = [gen, evals, f"{fitness.max():.3f}", f"{fitness.mean():.3f}",
               f"{np.median(fitness):.3f}", int(conns[b]),
               f"{conns.mean():.1f}", valid, f"{best_valid:.3f}", f"{dt:.2f}"]
        if v2:
            bs = BL.summarize_bar_stats(envs[0].bar_stats(), ecfg)
            row += [f"{bs['net_disp_mean']:.2f}", f"{bs['bars_moved_10']:.2f}",
                    f"{bs['push_frac']:.4f}", f"{bs['multi_push_bar_steps']:.1f}"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"gen {gen:4d} evals {evals:7d} best {fitness.max():6.2f} mean {fitness.mean():6.2f} "
              f"conns {conns[b]:5d}/{conns.mean():7.1f}" + (f" valid {valid}" if valid else "")
              + f" ({dt:.1f}s)", flush=True)
        if last:
            break
        genomes = next_generation(genomes, sel, args, rng)

    _save_pop(os.path.join(args.out, "final_pop.npz"), genomes, fitness)
    print(f"done in {time.perf_counter() - t_start:.0f}s; best validation {best_valid:.2f}")


def _save_pop(path, genomes, fitness):
    d = {"fitness": np.asarray(fitness)}
    for i, g in enumerate(genomes):
        d.update(g.to_dict(f"g{i}_"))
    np.savez_compressed(path, **d)


if __name__ == "__main__":
    main()
