import torch
import torch.nn as nn

from oprl.algos.utils import initialize_weight


class Critic(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.q1 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x)

    def Q1(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x)


class DoubleCritic(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.q1 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        self.q2 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x)


class MCCritic(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.q1 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        self.q2 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        self.q3 = MLP(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x), self.q2(x), self.q3(x)

    def Q1(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x)

    def get_action_grad(self, optim, states, actions): #, actions):
        q1, q2, q3 = self.forward(states, actions)
        q_cat = torch.cat((q1, q2, q3), dim=1).flatten()
        var = torch.var(q_cat)

        optim.zero_grad()
        var.backward(retain_graph=True)
        da = torch.autograd.grad(var, actions)

        return da[0] + 1e-8


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

        self.nn = nn.Sequential(*layers).apply(initialize_weight)

    def forward(self, x):
        return self.nn(x)

    def get_layer_norm(self):
        total_norm = 0
        for p in self.nn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
