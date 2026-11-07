
import subprocess
import torch

from run_episode import run_episode


def offspring_count(fitness):
    if fitness < 700:
        return 0
    elif fitness < 1000:
        return 1
    elif fitness < 1300:
        return 2
    elif fitness < 1500:
        return 3
    elif fitness < 1700:
        return 5
    elif fitness < 1900:
        return 6
    elif fitness < 2000:
        return 8
    elif fitness < 2200:
        return 9
    elif fitness <= 3000:
        return 10
    else:
        return 15


def reproduce(
    survivors,
    survivor_fitness,
    mutation_rate,
    mutation_scale,
    connection_mutation_rate,
    connection_mutation_scale
):
    new_genomes = []

    for parent, fitness in zip(survivors, survivor_fitness):
        count = offspring_count(fitness)

        for _ in range(count):
            child = parent.clone()

            child.mutate(
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale
            )

            if child.full_rank:
                child.add_remove_connection(
                    mutation_rate=connection_mutation_rate,
                    mutation_scale=connection_mutation_scale
                )

            new_genomes.append(child)

    return new_genomes


def serialize_genome(genome):

    if genome.full_rank:

        Win, Wh, Wout = genome.extract_weights()

        return {
            "full_rank": True,
            "Win": Win.to_sparse_csr(),
            "Wh": Wh.to_sparse_csr(),
            "Wout": Wout.to_sparse_csr(),
            "hyperparameters": (
                genome.extract_hyperparameters().clone()
            )
        }

    return {
        "full_rank": False,
        "genome": genome.clone()
    }


def evolve(genomes, generations=200):

    initial_gen = 11

    hall_of_fame = []

    for g in range(initial_gen, generations):

        episode_fitness = [
            run_episode(genomes, steps=300)
            for _ in range(6)
        ]

        fitness = [
            sum(episode[i] for episode in episode_fitness) / 6
            for i in range(len(genomes))
        ]

        ranking = sorted(
            range(len(genomes)),
            key=lambda i: fitness[i],
            reverse=True
        )

        hall_of_fame.extend([
            (fitness[i], genomes[i].clone())
            for i in ranking[:2]
        ])

        hall_of_fame.sort(
            key=lambda x: x[0],
            reverse=True
        )

        hall_of_fame = hall_of_fame[:20]

        survivors = [
            genomes[i]
            for i in ranking[:5]
        ]

        genomes = reproduce(
            survivors,
            offspring_counts=[6, 5, 4, 3, 1],
            mutation_rate=0.1,
            mutation_scale=0.05,
            connection_mutation_rate=0.001,
            connection_mutation_scale=1.0
        )

        for i, (_, genome) in enumerate(hall_of_fame):

            if i < len(genomes):
                genomes[i] = genome.clone()

        print(
            f"Generation {g}: "
            f"best={max(fitness):.2f}, "
            f"avg={sum(fitness)/len(fitness):.2f}"
        )

        if (g + 1) % 5 == 0:

            checkpoint = (
                f"checkpoint/generation_{g + 1}.pt"
            )

            subprocess.run(
                [
                    "git",
                    "pull",
                    "--rebase",
                    "origin",
                    "main"
                ],
                check=True
            )

            serialized_genomes = [
                serialize_genome(genome)
                for genome in genomes
            ]

            serialized_hall_of_fame = [
                (
                    fit,
                    serialize_genome(genome)
                )
                for fit, genome in hall_of_fame
            ]

            torch.save(
                {
                    "genomes": serialized_genomes,
                    "hall_of_fame": serialized_hall_of_fame
                },
                checkpoint
            )

            subprocess.run(
                ["git", "add", checkpoint],
                check=True
            )

            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Checkpoint generation {g + 1}"
                ],
                check=True
            )

            subprocess.run(
                [
                    "git",
                    "push",
                    "origin",
                    "main"
                ],
                check=True
            )

            print(
                f"Checkpoint pushed: generation {g + 1}"
            )

    return genomes, hall_of_fame
