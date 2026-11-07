import numpy as np


class ForagingEnv:
    """World mechanics, food, movement integration, and episode timing."""

    def __init__(
        self,
        agents,
        world_size=1000.0,
        n_food=200,
        food_radius=8.0,
        food_energy=1.0,
        sensor_range=150.0,
        max_speed=5.0,
        acceleration=1.0,
        friction=0.95,
        turn_rate=0.15,
        max_steps=500,
    ):
        self.agents = list(agents)

        if not self.agents:
            raise ValueError("At least one agent is required.")

        self.world_size = float(world_size)
        self.n_food = int(n_food)
        self.food_radius = float(food_radius)
        self.food_energy = float(food_energy)
        self.sensor_range = float(sensor_range)
        self.max_speed = float(max_speed)
        self.acceleration = float(acceleration)
        self.friction = float(friction)
        self.turn_rate = float(turn_rate)
        self.max_steps = int(max_steps)

        self.rng = np.random.default_rng()
        self.step_count = 0
        self.food_positions = np.empty((0, 2), dtype=np.float32)
        self.food_alive = np.empty(0, dtype=bool)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        margin = max(agent.radius for agent in self.agents)

        for agent in self.agents:
            position = self.rng.uniform(
                margin, self.world_size - margin, size=2
            )
            theta = self.rng.uniform(-np.pi, np.pi)
            agent.reset(position, theta)

        self.food_positions = self.rng.uniform(
            self.food_radius,
            self.world_size - self.food_radius,
            size=(self.n_food, 2),
        ).astype(np.float32)
        self.food_alive = np.ones(self.n_food, dtype=bool)

        return self._observations(), self._info()

    def _observations(self):
        n = len(self.agents)
        R = self.sensor_range

        positions = np.asarray(
            [a.position for a in self.agents], dtype=np.float32
        )
        thetas = np.asarray(
            [a.theta for a in self.agents], dtype=np.float32
        )
        obs = np.zeros((n, 16), dtype=np.float32)

        food = self.food_positions[self.food_alive]

        if len(food):
            delta = food[None, :, :] - positions[:, None, :]
            distances = np.linalg.norm(delta, axis=2)
            visible = (distances > 1e-6) & (distances <= R)

            angles = (
                np.arctan2(delta[:, :, 1], delta[:, :, 0])
                - thetas[:, None]
            ) % (2 * np.pi)
            sectors = (angles * (8 / (2 * np.pi))).astype(int) % 8
            strengths = 1.0 - distances / R

            for i in range(n):
                mask = visible[i]
                np.maximum.at(
                    obs[i, :8], sectors[i, mask], strengths[i, mask]
                )

        delta = positions[None, :, :] - positions[:, None, :]
        distances = np.linalg.norm(delta, axis=2)
        visible = (distances > 1e-6) & (distances <= R)

        angles = (
            np.arctan2(delta[:, :, 1], delta[:, :, 0])
            - thetas[:, None]
        ) % (2 * np.pi)
        sectors = (angles * (8 / (2 * np.pi))).astype(int) % 8
        strengths = 1.0 - distances / R

        for i in range(n):
            mask = visible[i]
            np.maximum.at(
                obs[i, 8:], sectors[i, mask], strengths[i, mask]
            )

        return obs

    def _info(self):
        return {
            "energy": np.array(
                [a.energy for a in self.agents], dtype=np.float32
            ),
            "food_remaining": int(self.food_alive.sum()),
            "steps": self.step_count,
        }

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (len(self.agents), 3)

        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected actions {expected_shape}; got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Actions must contain only finite values.")

        actions = np.clip(actions, -1.0, 1.0)
        rewards = np.zeros(len(self.agents), dtype=np.float32)

        positions = np.stack([a.position for a in self.agents])
        velocities = np.stack([a.velocity for a in self.agents])
        radii = np.array(
            [a.radius for a in self.agents], dtype=np.float32
        )
        thetas = np.array(
            [a.theta for a in self.agents], dtype=np.float32
        )

        thetas += actions[:, 2] * self.turn_rate
        cos_theta = np.cos(thetas)
        sin_theta = np.sin(thetas)

        ax = actions[:, 0] * cos_theta - actions[:, 1] * sin_theta
        ay = actions[:, 0] * sin_theta + actions[:, 1] * cos_theta

        acceleration = self.acceleration * np.column_stack((ax, ay))
        velocities = self.friction * velocities + acceleration

        speeds = np.linalg.norm(velocities, axis=1)
        too_fast = speeds > self.max_speed
        velocities[too_fast] *= (
            self.max_speed / speeds[too_fast]
        )[:, None]

        positions += velocities

        lower = radii[:, None]
        upper = self.world_size - radii[:, None]
        below = positions < lower
        above = positions > upper
        positions = np.clip(positions, lower, upper)

        velocities[below & (velocities < 0)] = 0
        velocities[above & (velocities > 0)] = 0

        for i, agent in enumerate(self.agents):
            agent.theta = float(thetas[i])
            agent.position[:] = positions[i]
            agent.velocity[:] = velocities[i]

        alive_indices = np.flatnonzero(self.food_alive)

        if len(alive_indices):
            agent_positions = np.stack(
                [a.position for a in self.agents]
            )
            agent_radii = np.array(
                [a.radius for a in self.agents]
            )

            delta = (
                agent_positions[:, None, :]
                - self.food_positions[None, alive_indices, :]
            )
            dist_sq = np.sum(delta * delta, axis=2)
            collision = dist_sq <= (
                agent_radii[:, None] + self.food_radius
            ) ** 2

            eligible_dist_sq = np.where(collision, dist_sq, np.inf)
            winners = np.argmin(eligible_dist_sq, axis=0)
            consumed = collision.any(axis=0)

            winning_agents = winners[consumed]
            consumed_food = alive_indices[consumed]

            rewards += (
                np.bincount(
                    winning_agents, minlength=len(self.agents)
                ).astype(np.float32)
                * self.food_energy
            )
            self.food_alive[consumed_food] = False

        for agent, reward in zip(self.agents, rewards):
            agent.energy += float(reward)

        self.step_count += 1
        terminated = not np.any(self.food_alive)
        truncated = self.step_count >= self.max_steps

        return (
            self._observations(),
            rewards,
            terminated,
            truncated,
            self._info(),
        )
