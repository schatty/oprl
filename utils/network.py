import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.l2_projection import _l2_project


def hidden_init(layer):
    fan_id = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_id)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model. """

    def __init__(self, state_size, action_size, action_bound_low, action_bound_high):
        """Initialize parameters and build model.
        Params:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        fc1_units = 400
        fc2_units = 300
        state_size = state_size[0]
        action_size = action_size[0]

        self.action_bound_low = float(action_bound_low)
        self.action_bound_high = float(action_bound_high)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions. """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        # Scale tanh output to lower and upper action bounds
        x = 0.5 * (x * (self.action_bound_high - self.action_bound_low) + (self.action_bound_high+self.action_bound_low))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, num_atoms, v_min, v_max):
        """Initialize parameters and build model.
        Params:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        fcs1_units = 400
        fc2_units = 300

        state_size = state_size[0]
        action_size = action_size[0]

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, num_atoms)
        self.reset_parameters()

        self.z_atoms = torch.linspace(v_min, v_max, num_atoms)

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # target_Z_dist - target Z distribution for next state
        # target_Z_atoms - atom values of target network with Bellman update applied

        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output_probs = F.softmax(x)

        return output_probs
