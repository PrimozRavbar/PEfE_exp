
import numpy as np
import torch

from brain import Brain
from brain2 import Brain2


class Agent:

    def __init__(self, genome, movement="continuous", brain_type="brain"):

        self.x = 0.0
        self.y = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.theta = 0.0
        self.omega = 0.0

        self.color = [1.0, 0.0, 0.0]

        self.energy = 0.0

        self.movement = movement

        if brain_type == "brain":

            self.brain = Brain(genome)

        elif brain_type == "brain2":

            self.brain = Brain2(genome)

        else:

            raise ValueError(
                f"Unknown brain type: {brain_type}"
            )


    def get_input(self, env):

        sensor_center = torch.tensor(
            [10.0, 0.0],
            dtype=torch.float32
        )

        theta = torch.tensor(
            self.theta,
            dtype=torch.float32
        )

        c = torch.cos(theta)
        s = torch.sin(theta)

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c])
        ])

        sensor_center = R @ sensor_center
        sensor_center[0] += self.x
        sensor_center[1] += self.y

        i, j = torch.meshgrid(
            torch.arange(10, dtype=torch.float32),
            torch.arange(10, dtype=torch.float32),
            indexing="ij"
        )

        local = torch.stack(
            (i - 4.5, j - 4.5),
            dim=-1
        )

        world = local @ R.T
        world += sensor_center

        x = torch.round(world[..., 0]).long()
        y = torch.round(world[..., 1]).long()

        valid = (
            (x >= 0) &
            (x < env.width) &
            (y >= 0) &
            (y < env.height)
        )

        inputs = torch.zeros(10, 10, 3)

        inputs[valid] = env.field[y[valid], x[valid]]

        return inputs.flatten()


    def get_polygon(self):

        corners = torch.tensor([
            [-5.0, -2.5],
            [ 5.0, -2.5],
            [ 5.0,  2.5],
            [-5.0,  2.5],
        ], dtype=torch.float32)

        theta = torch.tensor(
            self.theta,
            dtype=torch.float32
        )

        c = torch.cos(theta)
        s = torch.sin(theta)

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c])
        ])

        corners = corners @ R.T

        corners[:, 0] += self.x
        corners[:, 1] += self.y

        return corners


    def take_discrete_action(self, output):

        output = output[:9].detach().numpy()

        if np.max(output) > 0:

            output_rs = np.reshape(output, (3, 3))

            ind = np.where(
                output_rs == np.max(output_rs)
            )

            rand_ind = np.random.randint(
                0,
                np.shape(ind)[1]
            )

            move_row = ind[0][rand_ind] - 1
            move_col = ind[1][rand_ind] - 1

        else:

            move_row = 0
            move_col = 0

        return move_row, move_col


    def step(self, env, dt):

        inp = self.get_input(env)

        output = self.brain(inp)

        if self.movement == "discrete":

            move_row, move_col = self.take_discrete_action(output)

            self.x += move_col * dt
            self.y += move_row * dt

            if move_col != 0 or move_row != 0:
                self.theta = float(
                    np.arctan2(
                        move_row,
                        move_col
                    )
                )

        elif self.movement == "continuous":

            forward = output[0] - output[1]
            turn = output[2] - output[3]

            self.omega += turn.item() * dt

            self.omega *= 0.1

            self.theta += turn.item() * dt

            theta = torch.tensor(
                self.theta,
                dtype=torch.float32
            )

            ax = torch.cos(theta) * forward
            ay = torch.sin(theta) * forward

            self.vx += ax.item() * dt
            self.vy += ay.item() * dt

            friction = 0.9

            self.vx *= friction
            self.vy *= friction

            self.x += self.vx * dt
            self.y += self.vy * dt

        else:

            raise ValueError(
                f"Unknown movement algorithm: {self.movement}"
            )

        if self.x < 0:
            self.x = 0
            self.vx = 0.0

        elif self.x > env.width - 1:
            self.x = env.width - 1
            self.vx = 0.0

        if self.y < 0:
            self.y = 0
            self.vy = 0.0

        elif self.y > env.height - 1:
            self.y = env.height - 1
            self.vy = 0.0

        env.consume_food(self)
