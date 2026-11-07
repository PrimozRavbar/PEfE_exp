import torch
from matplotlib.path import Path


class Environment:

    def __init__(self):

        self.width = 500
        self.height = 500

        self.aura_sigma = 50.0

        # Static background
        self.background = torch.zeros(
            (self.height, self.width, 3),
            dtype=torch.float32
        )

        self.add_gaussians()

        # Blue 10-pixel border
        blue = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32
        )

        self.background[:10, :, :] = blue
        self.background[-10:, :, :] = blue
        self.background[:, :10, :] = blue
        self.background[:, -10:, :] = blue

        # Normalize existing green food to total volume 1000
        green = self.background[:, :, 1]
        green *= 25000.0 / green.sum()

        # Current world state
        self.field = self.background.clone()

    def add_gaussians(self):

        y, x = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing="ij"
        )

        for _ in range(4):

            cx = torch.randint(0, self.width, (1,)).item()
            cy = torch.randint(0, self.height, (1,)).item()
            sigma = torch.empty(1).uniform_(30.0, 30.0).item()

            gaussian = torch.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) /
                (2 * sigma ** 2)
            )

            self.background[:, :, 1] += gaussian


    def update(self, agents):

        # Restore static world
        self.field.copy_(self.background)

        # Draw aura first, then agent body
        for agent in agents:
            self.draw_aura(agent)

        for agent in agents:
            self.draw_agent(agent)


    def draw_aura(self, agent):

        radius = int(self.aura_sigma * 3)

        xmin = max(int(agent.x) - radius, 0)
        xmax = min(int(agent.x) + radius + 1, self.width)

        ymin = max(int(agent.y) - radius, 0)
        ymax = min(int(agent.y) + radius + 1, self.height)

        if xmin >= xmax or ymin >= ymax:
            return

        yy, xx = torch.meshgrid(
            torch.arange(ymin, ymax, dtype=torch.float32),
            torch.arange(xmin, xmax, dtype=torch.float32),
            indexing="ij"
        )

        gaussian = 0.1*torch.exp(
            -((xx - agent.x) ** 2 + (yy - agent.y) ** 2) /
            (2 * self.aura_sigma ** 2)
        )

        color = torch.tensor(agent.color, dtype=torch.float32)

        self.field[ymin:ymax, xmin:xmax] += (
            gaussian.unsqueeze(-1) * color
        )

        self.field[ymin:ymax, xmin:xmax].clamp_(0.0, 1.0)

    def draw_agent(self, agent):

        polygon = agent.get_polygon().numpy()

        xmin = max(int(polygon[:, 0].min()) - 1, 0)
        xmax = min(int(polygon[:, 0].max()) + 2, self.width)

        ymin = max(int(polygon[:, 1].min()) - 1, 0)
        ymax = min(int(polygon[:, 1].max()) + 2, self.height)

        if xmin >= xmax or ymin >= ymax:
            return

        yy, xx = torch.meshgrid(
            torch.arange(ymin, ymax),
            torch.arange(xmin, xmax),
            indexing="ij"
        )

        points = torch.stack(
            (xx.flatten(), yy.flatten()),
            dim=1
        ).numpy()

        mask = Path(polygon).contains_points(points)
        mask = torch.from_numpy(
            mask.reshape(ymax - ymin, xmax - xmin)
        )

        self.field[ymin:ymax, xmin:xmax][mask] = torch.tensor(
            agent.color,
            dtype=torch.float32
        )

    def consume_food(self, agent):

        x0 = max(int(agent.x) - 5, 0)
        x1 = min(int(agent.x) + 5, self.width)

        y0 = max(int(agent.y) - 5, 0)
        y1 = min(int(agent.y) + 5, self.height)

        food = self.background[y0:y1, x0:x1, 1]

        amount = food.sum().item()

        agent.energy += amount

        food.zero_()
