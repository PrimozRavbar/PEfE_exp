import torch


def offspring_count(fitness):
    if fitness < 0.5:
        return 0
    return max(1, int(fitness / 5))
def reproduce(
    survivors,
    survivor_fitness,
    mutation_rate,
    mutation_scale,
    connection_mutation_rate,
    connection_mutation_scale,
):
    new_genomes = []

    for parent, fitness in zip(survivors, survivor_fitness):
        count = offspring_count(fitness)

        for _ in range(count):
            child = parent.clone()
            child.mutate(
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
            )

            if child.full_rank:
                child.add_remove_connection(
                    mutation_rate=connection_mutation_rate,
                    mutation_scale=connection_mutation_scale,
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
            ),
        }

    return {
        "full_rank": False,
        "genome": genome.clone(),
    }
