# PEEvo

Experimenting with parameter-efficient evolution of recurrent neural networks embedded in simple 2D agents that interact with their environment and with one another.

## Source

- `environment.py` — 2D environment, food distribution, agent rendering, and food consumption.
- `brain.py` — Recurrent neural network and parameter-efficient weight generation.
- `agent.py` — Agent sensing, movement, orientation, and interaction with the environment.
- `genome.py` — Genome representation, mutation, and parameter-efficient network weights.
- `run_episode.py` — Runs individual simulation episodes and optionally renders them.
- `evolution.py` — Population evaluation, selection, reproduction, and evolutionary checkpoints.
