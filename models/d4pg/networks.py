import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, num_states, num_actions, hidden_size, v_min, v_max,
                 num_atoms, init_w=3e-3, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): size of the hidden layers
            v_min (float): minimum value for critic
            v_max (float): maximum value for critic
            num_atoms (int): number of atoms in distribution
            init_w:
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_states + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_atoms)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_probs(self, state, action):
        return F.softmax(self.forward(state, action), dim=1)


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, init_w=3e-3, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int):  action dimension
            hidden_size (int): size of the hidden layer
            init_w:
        """
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))

        return x

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action