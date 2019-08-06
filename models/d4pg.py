import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import OUNoise, ReplayBuffer
from utils.l2_projection import _l2_project
from env.utils import create_env_wrapper


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
        return F.softmax(self.forward(state, action))


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

        x = 0.5 * x * (2 - (-2))

        return x

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy().item()


class LearnerD4PG(object):
    """Policy and value network update routine. """

    def __init__(self, config, batch_queue):
        hidden_dim = config['dense_size']
        state_dim = config['state_dims'][0]
        action_dim = config['action_dims'][0]
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        v_min = config['v_min']
        v_max = config['v_max']
        num_atoms = config['num_atoms']
        self.device = config['device']
        self.max_frames = config['num_episodes_train']
        self.max_steps = config['max_ep_length']
        self.frame_idx = 0
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.gamma = config['discount_rate']
        self.batch_queue = batch_queue

        # Noise process
        env = create_env_wrapper(config)
        self.ou_noise = OUNoise(env.get_action_space())
        del env

        # Value and policy nets
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim, v_min, v_max, num_atoms, device=self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim, v_min, v_max, num_atoms, device=self.device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.BCEWithLogitsLoss()

    def ddpg_update(self, batch, min_value=-np.inf, max_value=np.inf):
        state, action, reward, next_state, done = batch

        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().unsqueeze(1).to(self.device)
        done = torch.from_numpy(done).float().unsqueeze(1).to(self.device)

        # ------- Update critic -------

        # Predict next actions with target policy network
        next_action = self.target_policy_net(next_state)

        # Predict Z distribution with target value network
        target_value = self.target_value_net.get_probs(next_state, next_action.detach())
        target_z_atoms = self.value_net.z_atoms

        # Batch of z-atoms
        target_Z_atoms = np.repeat(np.expand_dims(target_z_atoms, axis=0), self.batch_size,
                                   axis=0)  # [batch_size x n_atoms]
        # Value of terminal states is 0 by definition
        target_Z_atoms *= (done.cpu().int().numpy() == 0)

        # Apply bellman update to each atom (expected value)
        reward = reward.cpu().float().numpy()
        target_Z_atoms = reward + (target_Z_atoms * self.gamma)
        target_z_projected = _l2_project(torch.from_numpy(target_Z_atoms).cpu().float(),
                                         target_value.cpu().float(),
                                         torch.from_numpy(self.value_net.z_atoms).cpu().float())

        critic_value = self.value_net(state, action)
        critic_value = critic_value.to(self.device)
        value_loss = self.value_criterion(critic_value,
                                     torch.autograd.Variable(target_z_projected, requires_grad=False).cuda())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------- Update actor -----------

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def run(self, stop_agent_event):
        while not self.batch_queue.empty() or not stop_agent_event.value:
            if self.batch_queue.empty():
                continue
            batch = list(zip(*self.batch_queue.get()))
            self.ddpg_update(batch)
        print("Exit learner.")
