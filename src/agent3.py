import numpy as np


class Agent:
    """Agent state and sensory interface."""

    def __init__(self, radius=5.0, brain=None):
        self.radius = float(radius)
        self.brain = brain

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.theta = 0.0
        self.energy = 0.0

        self.n_sensors = 8

    def reset(self, position, theta):
        self.position[:] = position
        self.velocity.fill(0)
        self.theta = float(theta)
        self.energy = 0.0

    def observe(self, env):
        """Return 8 food sensors + 8 other-agent sensors."""
        obs = np.zeros(16, dtype=np.float32)
        R = env.sensor_range

        food = env.food_positions[env.food_alive]

        if len(food):
            delta = food - self.position
            distances = np.linalg.norm(delta, axis=1)
            visible = (distances > 1e-6) & (distances <= R)

            if np.any(visible):
                angles = (
                    np.arctan2(delta[visible, 1], delta[visible, 0])
                    - self.theta
                ) % (2 * np.pi)
                sectors = (angles * (8 / (2 * np.pi))).astype(int) % 8
                strengths = 1.0 - distances[visible] / R
                np.maximum.at(obs[:8], sectors, strengths)

        positions = np.asarray(
            [other.position for other in env.agents if other is not self],
            dtype=np.float32,
        )

        if len(positions):
            delta = positions - self.position
            distances = np.linalg.norm(delta, axis=1)
            visible = (distances > 1e-6) & (distances <= R)

            if np.any(visible):
                angles = (
                    np.arctan2(delta[visible, 1], delta[visible, 0])
                    - self.theta
                ) % (2 * np.pi)
                sectors = (angles * (8 / (2 * np.pi))).astype(int) % 8
                strengths = 1.0 - distances[visible] / R
                np.maximum.at(obs[8:], sectors, strengths)

        return obs
