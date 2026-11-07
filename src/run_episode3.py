import time
import numpy as np
import torch


def run_episode(env, renderer=None, seed=None, max_steps=None):
    if seed is None:
        observations, info = env.reset()
    else:
        observations, info = env.reset(seed=seed)

    limit = env.max_steps if max_steps is None else int(max_steps)

    initial_energy = np.array(
        [agent.energy for agent in env.agents],
        dtype=np.float64,
    )

    if renderer is not None:
        renderer.frames.clear()
        renderer.capture(env)

    start = time.perf_counter()

    for _ in range(limit):
        actions = []

        for agent, observation in zip(env.agents, observations):
            if agent.brain is None:
                raise ValueError("Every agent must have a brain.")

            sensory = torch.as_tensor(
                observation, dtype=torch.float32
            )

            with torch.no_grad():
                output = agent.brain(sensory)

            actions.append(output[:3].detach().cpu().numpy())

        actions = np.asarray(actions, dtype=np.float32)

        observations, rewards, terminated, truncated, info = env.step(
            actions
        )

        if renderer is not None:
            renderer.capture(env)

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - start

    energy = np.array(
        [agent.energy for agent in env.agents],
        dtype=np.float64,
    )

    food_collected = energy - initial_energy

    return {
        "steps": env.step_count,
        "energy": energy,
        "food_collected": food_collected,
        "food_remaining": int(env.food_alive.sum()),
        "seconds": elapsed,
    }
