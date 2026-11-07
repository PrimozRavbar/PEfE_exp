
import torch


INPUT_SIZE = 300
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

R_IN = 18
R_H = 13
R_OUT = 2


MAX_WIN_PARAMETERS = (
    HIDDEN_SIZE * INPUT_SIZE
) // 10

MAX_WH_PARAMETERS = (
    HIDDEN_SIZE * HIDDEN_SIZE
) // 10

MAX_WOUT_PARAMETERS = (
    OUTPUT_SIZE * HIDDEN_SIZE
) // 10


NUM_PARAMETERS_PE = (
    HIDDEN_SIZE * R_IN + R_IN * INPUT_SIZE +
    HIDDEN_SIZE * R_H + R_H * HIDDEN_SIZE +
    OUTPUT_SIZE * R_OUT + R_OUT * HIDDEN_SIZE
)

NUM_PARAMETERS_FULL = (
    HIDDEN_SIZE * INPUT_SIZE +
    HIDDEN_SIZE * HIDDEN_SIZE +
    OUTPUT_SIZE * HIDDEN_SIZE
)

NUM_HYPERPARAMETERS = 20


class GenomePE:

    def __init__(self, full_rank=False, initialization="random"):

        self.full_rank = full_rank

        if full_rank:

            if initialization == "zeros":

                weights = torch.zeros(NUM_PARAMETERS_FULL)

            elif initialization == "random":

                weights = torch.randn(NUM_PARAMETERS_FULL) * 0.1

            else:

                raise ValueError(
                    "initialization must be 'zeros' or 'random'"
                )

            self.chromosome = torch.cat([
                weights,
                torch.ones(NUM_HYPERPARAMETERS)
            ])

        else:

            Ain = torch.zeros(HIDDEN_SIZE, R_IN)
            Bin = torch.randn(R_IN, INPUT_SIZE) * 0.1

            Ah = torch.zeros(HIDDEN_SIZE, R_H)
            Bh = torch.randn(R_H, HIDDEN_SIZE) * 0.1

            Aout = torch.zeros(OUTPUT_SIZE, R_OUT)
            Bout = torch.randn(R_OUT, HIDDEN_SIZE) * 0.1

            self.chromosome = torch.cat([
                Ain.flatten(),
                Bin.flatten(),
                Ah.flatten(),
                Bh.flatten(),
                Aout.flatten(),
                Bout.flatten(),
                torch.ones(NUM_HYPERPARAMETERS)
            ])

    def clone(self):

        g = GenomePE(
            full_rank=self.full_rank
        )

        g.chromosome = self.chromosome.clone()

        return g

    def mutate(
        self,
        mutation_rate=0.1,
        mutation_scale=0.05
    ):

        if self.full_rank:

            weights = self.chromosome[
                :NUM_PARAMETERS_FULL
            ]

            mask = (
                (weights != 0)
                & (
                    torch.rand(NUM_PARAMETERS_FULL)
                    < mutation_rate
                )
            )

            self.chromosome[
                :NUM_PARAMETERS_FULL
            ] += (
                mask
                * torch.randn(NUM_PARAMETERS_FULL)
                * mutation_scale
            )

            hyperparameters = self.chromosome[
                NUM_PARAMETERS_FULL:
            ]

            hyper_mask = (
                torch.rand(NUM_HYPERPARAMETERS)
                < mutation_rate
            )

            hyperparameters += (
                hyper_mask
                * torch.randn(NUM_HYPERPARAMETERS)
                * mutation_scale
            )

        else:

            mask = (
                torch.rand_like(self.chromosome)
                < mutation_rate
            )

            self.chromosome += (
                mask
                * torch.randn_like(self.chromosome)
                * mutation_scale
            )

    def add_remove_connection(
        self,
        mutation_rate=0.1,
        mutation_scale=0.05
    ):

        if not self.full_rank:
            return

        matrix_ranges = [

            (
                0,
                HIDDEN_SIZE * INPUT_SIZE,
                MAX_WIN_PARAMETERS
            ),

            (
                HIDDEN_SIZE * INPUT_SIZE,
                HIDDEN_SIZE * INPUT_SIZE
                + HIDDEN_SIZE * HIDDEN_SIZE,
                MAX_WH_PARAMETERS
            ),

            (
                HIDDEN_SIZE * INPUT_SIZE
                + HIDDEN_SIZE * HIDDEN_SIZE,
                NUM_PARAMETERS_FULL,
                MAX_WOUT_PARAMETERS
            )
        ]

        for start, end, max_parameters in matrix_ranges:

            weights = self.chromosome[start:end]

            mask = (
                torch.rand(end - start)
                < mutation_rate
            )

            remove = torch.nonzero(
                mask & (weights != 0),
                as_tuple=False
            ).flatten()

            add = torch.nonzero(
                mask & (weights == 0),
                as_tuple=False
            ).flatten()

            for i in remove:

                weights[i] = 0.0

            current_count = (
                weights != 0
            ).sum().item()

            available = (
                max_parameters
                - current_count
            )

            if available > 0:

                if len(add) > available:

                    add = add[
                        torch.randperm(len(add))[
                            :available
                        ]
                    ]

                for i in add:

                    weights[i] = (
                        torch.randn(1).item()
                        * mutation_scale
                    )

            self.chromosome[start:end] = weights

    def extract_weights(self):

        if self.full_rank:

            idx = 0

            n_Win = (
                HIDDEN_SIZE * INPUT_SIZE
            )

            n_Wh = (
                HIDDEN_SIZE * HIDDEN_SIZE
            )

            n_Wout = (
                OUTPUT_SIZE * HIDDEN_SIZE
            )

            Win = self.chromosome[
                idx:idx + n_Win
            ].reshape(
                HIDDEN_SIZE,
                INPUT_SIZE
            )

            idx += n_Win

            Wh = self.chromosome[
                idx:idx + n_Wh
            ].reshape(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            )

            idx += n_Wh

            Wout = self.chromosome[
                idx:idx + n_Wout
            ].reshape(
                OUTPUT_SIZE,
                HIDDEN_SIZE
            )

            return Win, Wh, Wout

        pe_chromosome = self.chromosome[
            :NUM_PARAMETERS_PE
        ]

        idx = 0

        n_Ain = HIDDEN_SIZE * R_IN
        n_Bin = R_IN * INPUT_SIZE

        n_Ah = HIDDEN_SIZE * R_H
        n_Bh = R_H * HIDDEN_SIZE

        n_Aout = OUTPUT_SIZE * R_OUT
        n_Bout = R_OUT * HIDDEN_SIZE

        Ain = pe_chromosome[
            idx:idx + n_Ain
        ].reshape(
            HIDDEN_SIZE,
            R_IN
        )

        idx += n_Ain

        Bin = pe_chromosome[
            idx:idx + n_Bin
        ].reshape(
            R_IN,
            INPUT_SIZE
        )

        idx += n_Bin

        Ah = pe_chromosome[
            idx:idx + n_Ah
        ].reshape(
            HIDDEN_SIZE,
            R_H
        )

        idx += n_Ah

        Bh = pe_chromosome[
            idx:idx + n_Bh
        ].reshape(
            R_H,
            HIDDEN_SIZE
        )

        idx += n_Bh

        Aout = pe_chromosome[
            idx:idx + n_Aout
        ].reshape(
            OUTPUT_SIZE,
            R_OUT
        )

        idx += n_Aout

        Bout = pe_chromosome[
            idx:idx + n_Bout
        ].reshape(
            R_OUT,
            HIDDEN_SIZE
        )

        Win = Ain @ Bin
        Wh = Ah @ Bh
        Wout = Aout @ Bout

        return Win, Wh, Wout

    def extract_hyperparameters(self):

        offset = (
            NUM_PARAMETERS_FULL
            if self.full_rank
            else NUM_PARAMETERS_PE
        )

        return self.chromosome[
            offset:
            offset + NUM_HYPERPARAMETERS
        ]
