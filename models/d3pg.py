import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import OUNoise, ReplayBuffer
from env.utils import create_env_wrapper


class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, num_states, num_actions, hidden_size, init_w=3e-3, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): number of neurons in hidden layers
            init_w:
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_states + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, init_w=3e-3, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (action): action dimension
            hidden_size (int): hidden size dimension
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
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy().item()


class LearnerD3PG(object):
    """Policy and value network update routine. """

    def __init__(self, config, batch_queue):
        """
        Args:
            config (dict): configuration
            batch_queue (multiproc.queue): queue with batches of replays
        """
        hidden_dim = config['dense_size']
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        state_dim = config['state_dims'][0]
        action_dim = config['action_dims'][0]
        self.device = config['device']
        self.max_frames = config['num_episodes_train']
        self.max_steps = config['max_ep_length']
        self.frame_idx = 0
        self.batch_size = config['batch_size']
        self.batch_queue = batch_queue
        self.gamma = config['discount_rate']
        self.tau = config['tau']

        # Noise process
        env = create_env_wrapper(config)
        self.ou_noise = OUNoise(env.get_action_space())
        del env

        # Value and policy nets
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss()

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

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------- Update actor --------

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update
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
                #print("Continuing as batch queue empty")
                continue
            batch = list(zip(*self.batch_queue.get()))
            self.ddpg_update(batch)
        print("Exit learner.")
