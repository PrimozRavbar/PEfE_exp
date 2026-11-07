"""Simple generational evolution of Brain3 foraging agents.

Each generation the population is shuffled into arenas of `--arena-size`
agents (default 10, 200 food, 1000 steps — the notebook's test setting),
every genome's fitness is its mean food collected over `--episodes`
shuffled episodes, and the next generation is built by rank-based
truncation selection with elitism (fixed population size).

evolution3.offspring_count is not used: it gives zero offspring below
fitness 0.5, so a zero-initialized population would go extinct at once.

Usage (from src/):
    python evolve3_zero.py --init zeros --generations 100
    python evolve3_zero.py --init random --target 15
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from agent3 import Agent
from brain3 import Brain3
from environment3 import ForagingEnv
from evolution3 import serialize_genome
from genome_pe3 import GenomePE, NUM_PARAMETERS_FULL
from run_episode3 import run_episode


def evaluate(genomes, args, rng, episodes):
    """Mean food per genome over `episodes` shuffled arena assignments."""
    n = len(genomes)
    totals = np.zeros(n)

    for _ in range(episodes):
        order = rng.permutation(n)
        for start in range(0, n, args.arena_size):
            idx = order[start:start + args.arena_size]
            # Fresh brains each episode so hidden state does not carry over.
            agents = [Agent(brain=Brain3(genomes[i])) for i in idx]
            env = ForagingEnv(agents, n_food=args.food, max_steps=args.steps)
            result = run_episode(env, seed=int(rng.integers(2**31)))
            totals[idx] += result["food_collected"]

    return totals / episodes


def team_score(genome, args, rng, episodes):
    """Mean food per agent for an arena of clones of one genome."""
    scores = []
    for _ in range(episodes):
        agents = [Agent(brain=Brain3(genome)) for _ in range(args.arena_size)]
        env = ForagingEnv(agents, n_food=args.food, max_steps=args.steps)
        result = run_episode(env, seed=int(rng.integers(2**31)))
        scores.append(result["food_collected"].mean())
    return float(np.mean(scores)), float(np.std(scores))


def n_connections(genome):
    return int((genome.chromosome[:NUM_PARAMETERS_FULL] != 0).sum().item())


def next_generation(genomes, fitness, args, rng):
    # Random tie-break so a flat (all-zero) fitness landscape still mixes.
    order = np.lexsort((rng.random(len(genomes)), -fitness))
    survivors = [genomes[i] for i in order[:args.survivors]]

    new = [g.clone() for g in survivors[:args.elites]]

    # Rank-weighted parent choice: weight k for rank 1, k-1 for rank 2, ...
    weights = np.arange(len(survivors), 0, -1, dtype=float)
    weights /= weights.sum()

    while len(new) < args.population:
        parent = survivors[rng.choice(len(survivors), p=weights)]
        child = parent.clone()
        child.mutate(
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
        )
        child.add_remove_connection(
            mutation_rate=args.connection_rate,
            mutation_scale=args.connection_scale,
        )
        new.append(child)

    return new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=["zeros", "random"], default="zeros")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--arena-size", type=int, default=10)
    p.add_argument("--food", type=int, default=200)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--survivors", type=int, default=8)
    p.add_argument("--elites", type=int, default=3)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--mutation-rate", type=float, default=0.1)
    p.add_argument("--mutation-scale", type=float, default=0.05)
    p.add_argument("--connection-rate", type=float, default=0.002)
    p.add_argument("--connection-scale", type=float, default=1.0)
    p.add_argument("--target", type=float, default=None,
                   help="stop when a clone team of the best genome "
                        "averages at least this much food per agent")
    p.add_argument("--check-every", type=int, default=5)
    p.add_argument("--check-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/evolve3")
    p.add_argument("--checkpoint-every", type=int, default=10)
    args = p.parse_args()

    if args.population % args.arena_size:
        p.error("--population must be a multiple of --arena-size")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    genomes = [
        GenomePE(full_rank=True, initialization=args.init)
        for _ in range(args.population)
    ]

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "generation", "best", "mean", "median",
            "best_connections", "mean_connections", "team_score", "seconds",
        ])

    best_team = (-1.0, None)

    for gen in range(args.generations):
        t0 = time.perf_counter()
        fitness = evaluate(genomes, args, rng, args.episodes)
        conns = np.array([n_connections(g) for g in genomes])
        b = int(np.argmax(fitness))

        team = ""
        if args.target is not None and (gen + 1) % args.check_every == 0:
            score, sd = team_score(genomes[b], args, rng, args.check_episodes)
            team = f"{score:.2f}"
            if score > best_team[0]:
                best_team = (score, genomes[b].clone())
                torch.save(
                    {"team_score": score, "team_sd": sd, "generation": gen,
                     "genome": serialize_genome(genomes[b])},
                    os.path.join(args.out, "best.pt"),
                )

        genomes_next = next_generation(genomes, fitness, args, rng)
        dt = time.perf_counter() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                gen, f"{fitness.max():.2f}", f"{fitness.mean():.3f}",
                f"{np.median(fitness):.2f}", int(conns[b]),
                f"{conns.mean():.1f}", team, f"{dt:.1f}",
            ])
        print(
            f"gen {gen:4d}  best={fitness.max():6.2f}  "
            f"mean={fitness.mean():6.2f}  "
            f"best_conns={conns[b]:5d}  mean_conns={conns.mean():7.1f}"
            + (f"  team={team}" if team else "")
            + f"  ({dt:.1f}s)",
            flush=True,
        )

        if (gen + 1) % args.checkpoint_every == 0:
            torch.save(
                {"generation": gen + 1,
                 "genomes": [serialize_genome(g) for g in genomes],
                 "fitness": fitness.tolist()},
                os.path.join(args.out, f"generation_{gen + 1}.pt"),
            )

        if args.target is not None and best_team[0] >= args.target:
            print(f"TARGET REACHED: clone team averages {best_team[0]:.2f} "
                  f"food/agent at generation {gen}", flush=True)
            break

        genomes = genomes_next

    print(f"best clone-team score: {best_team[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
