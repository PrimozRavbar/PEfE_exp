
import torch
import imageio.v3 as iio

from environment import Environment
from agent import Agent


def run_episode(
    genomes,
    steps=500,
    render=False,
    movement="continuous",
    brain_type="brain",
    record_data=False
):

    env = Environment()

    agents = []

    for genome in genomes:

        agent = Agent(
            genome,
            movement=movement,
            brain_type=brain_type
        )

        agent.x = torch.randint(
            0,
            env.width,
            (1,)
        ).item()

        agent.y = torch.randint(
            0,
            env.height,
            (1,)
        ).item()

        agent.theta = (
            2 * torch.pi * torch.rand(1).item()
        )

        agents.append(agent)

    frames = []

    if record_data:

        recorded_data = []

        for agent in agents:

            recorded_data.append({
                "sensory": [],
                "hidden": [],
                "output": [],
                "Win": agent.brain.Win.detach().clone(),
                "Wh": agent.brain.Wh.detach().clone(),
                "Wout": agent.brain.Wout.detach().clone()
            })

    for t in range(steps):

        env.update(agents)

        for i, agent in enumerate(agents):

            agent.step(
                env,
                dt=1.0
            )

            if record_data:

                recorded_data[i]["sensory"].append(
                    agent.brain.last_sensory.clone()
                )

                recorded_data[i]["hidden"].append(
                    agent.brain.last_hidden.clone()
                )

                recorded_data[i]["output"].append(
                    agent.brain.last_output.clone()
                )

        if render:

            frames.append(
                (255 * env.field)
                .clamp(0, 255)
                .byte()
                .numpy()
            )

    if render:

        iio.imwrite(
            "simulation.mp4",
            frames,
            fps=20
        )

    fitness = [
        agent.energy
        for agent in agents
    ]

    if record_data:

        for data in recorded_data:

            data["sensory"] = torch.stack(
                data["sensory"]
            )

            data["hidden"] = torch.stack(
                data["hidden"]
            )

            data["output"] = torch.stack(
                data["output"]
            )

        return fitness, recorded_data

    return fitness
