from copy import deepcopy

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(), output_activation=nn.Identity()):
        super().__init__()

        layers = []
        units = input_dim
        for next_units in hidden_units:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        layers.append(nn.Linear(units, output_dim))
        layers.append(output_activation)

        self.nn = nn.Sequential(*layers)# .apply(initialize_weight)

    def forward(self, x):
        return self.nn(x)

    def get_layer_norm(self):
        total_norm = 0
        for p in self.nn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


class DeterministicPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )#.apply(initialize_weight)

    def forward(self, states):
        return torch.tanh(self.mlp(states))

    def exploit(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.forward(state).cpu().data.numpy().flatten()
