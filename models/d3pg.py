import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import OUNoise, ReplayBuffer, NormalizedActions


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.to('cuda')

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda')
        action = self.forward(state)
        return action.detach().cpu().numpy().item()


class LearnerD3PG(object):
    def __init__(self, config, batch_queue):
        self.batch_queue = batch_queue
        self.env = NormalizedActions(gym.make("Pendulum-v0"))
        self.ou_noise = OUNoise(self.env.action_space)
        device = 'cuda:0'

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = config['dense1_size']

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        value_lr = 1e-3
        policy_lr = 1e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss()

        self.max_frames = config['num_episodes_train']
        self.max_steps = 1000
        self.frame_idx = 0
        self.batch_size = config['batch_size']

    def ddpg_update(self, batch, batch_size, gamma=0.99, min_value=-np.inf, max_value=np.inf, soft_tau=1e-2):
        state, action, reward, next_state, done = batch

        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)

        state = torch.from_numpy(state).float().to('cuda')
        next_state = torch.from_numpy(next_state).float().to('cuda')
        action = torch.from_numpy(action).float().to('cuda')
        reward = torch.from_numpy(reward).float().unsqueeze(1).to('cuda')
        done = torch.from_numpy(done).float().unsqueeze(1).to('cuda')

        # Update critic

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update actor

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def run(self, stop_agent_event):
        while not self.batch_queue.empty() or not stop_agent_event.value:
            if self.batch_queue.empty():
                #print("Continuing as batch queue empty")
                continue
            batch = list(zip(*self.batch_queue.get()))
            self.ddpg_update(batch, self.batch_size)
        print("Exit learner.")

