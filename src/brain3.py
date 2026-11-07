import torch
import torch.nn as nn


class Brain3(nn.Module):
    def __init__(
        self, genome, input_size=16, hidden_size=128, output_size=3
    ):
        super().__init__()

        self.Win, self.Wh, self.Wout = genome.extract_weights()
        self.h = torch.zeros(hidden_size)

        B_hyper_param = genome.extract_hyperparameters()
        self.hidden_max = B_hyper_param[0] * 10000
        self.add_hidden = B_hyper_param[1] * 0.5
        self.add_sens = B_hyper_param[2] * 0.01
        self.leak_sens = B_hyper_param[3] * 0.1
        self.leak_hidden = B_hyper_param[4] * 0.1
        self.max_epsilon = B_hyper_param[5] * 1

        self.last_sensory = None
        self.last_hidden = None
        self.last_output = None
        self.last_epsilon = None

    def forward(self, x):
        x = x.view(-1)

        sensory = self.Win @ x
        sensory = torch.clamp(sensory, 0, self.hidden_max)

        recurrent = self.Wh @ self.h

        self.h = self.h - self.leak_hidden * self.h
        self.h = torch.clamp(self.h, 0, self.hidden_max)

        self.h = (
            self.h
            + self.add_sens * sensory
            + self.add_hidden * recurrent
        )
        self.h = torch.clamp(self.h, 0, self.hidden_max)

        self.out_layer = self.Wout @ self.h

        epsilon = self.h[79] * 100
        epsilon = torch.clamp(epsilon, 0, self.max_epsilon)

        noise = torch.randn_like(self.out_layer) * self.out_layer * epsilon
        output = self.out_layer + noise
        #output = torch.clamp(output, min=0)
        output = torch.tanh(output)

        self.last_sensory = sensory.detach().clone()
        self.last_hidden = self.h.detach().clone()
        self.last_output = output.detach().clone()
        self.last_epsilon = epsilon.detach().clone()

        return output
