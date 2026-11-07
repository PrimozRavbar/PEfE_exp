import torch
import torch.nn as nn


class Brain(nn.Module):

    def __init__(self, genome,
                 input_size=300,
                 hidden_size=128,
                 output_size=10):

        super().__init__()

        self.Win, self.Wh, self.Wout = genome.extract_weights()

        self.h = torch.zeros(hidden_size)

        self.leak = 0.6
        self.hidden_max = 10000
        self.add_hidden = 0.5
        self.add_sens = 0.01
        self.output_gain = 100

        self.last_sensory = None
        self.last_hidden = None
        self.last_output = None

    def forward(self, x):

        # Sensory input
        sensory = self.Win @ x

        # Clamp sensory drive
        sensory = torch.clamp(
            sensory,
            0,
            self.hidden_max
        )

        # Recurrent drive
        recurrent = self.Wh @ self.h

        # Leak hidden state
        self.h = (1.0 - self.leak) * self.h

        # Add sensory and recurrent contributions
        self.h = (
            self.h
            + self.add_sens * sensory
            + self.add_hidden * recurrent
        )

        # Clamp hidden activity
        self.h = torch.clamp(
            self.h,
            0,
            self.hidden_max
        )

        # Output
        output = self.Wout @ self.h

        # Positive motor neurons
        output = torch.clamp(
            output,
            min=0
        ) * self.output_gain

        # Store activations from this forward pass
        self.last_sensory = sensory.detach().clone()
        self.last_hidden = self.h.detach().clone()
        self.last_output = output.detach().clone()

        return output
