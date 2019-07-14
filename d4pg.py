import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    z_p = torch.tensor(z_p).float()

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]

    d_pos = torch.cat([z_q, vmin[None]], 0)[1:]
    d_neg = torch.cat([vmax[None], z_q], 0)[:-1]

    # Clip z_p to be in new support range (vmin, vmax)
    z_p = torch.clamp(z_p, vmin, vmax)[:, None, :]

    # Get the distance between atom values in support
    d_pos = (d_pos - z_q)[None, :, None]
    d_neg = (z_q - d_neg)[None, :, None]
    z_q = z_q[None, :, None]

    d_neg = torch.where(d_neg>0, 1./d_neg, torch.zeros(d_neg.shape))
    d_pos = torch.where(d_pos>0, 1./d_pos, torch.zeros(d_pos.shape))

    delta_qp = z_p - z_q
    d_sign = (delta_qp >= 0).type(p.dtype)

    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]
    return torch.sum(torch.clamp(1.-delta_hat, 0., 1.) * p, -1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1 ) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10_000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f"frame {frame_idx}. reward {rewards[-1]}")
    plt.plot(rewards)
    #plt.show()


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 51)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        v_min = -20.0
        v_max = 0.0
        num_atoms = 51
        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        output_probs = F.softmax(x)
        return output_probs

    def get_probs(self, state, action):
        return F.softmax(self.forward(state, action))


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda')
        action = self.forward(state)
        return action.detach().cpu().numpy().item()


def ddpg_update(replay_buffer, batch_size, gamma=0.99, min_value=-np.inf, max_value=np.inf, soft_tau=1e-3):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.from_numpy(state).float().to('cuda')
    next_state = torch.from_numpy(next_state).float().to('cuda')
    action = torch.from_numpy(action).float().to('cuda')
    reward = torch.from_numpy(reward).float().unsqueeze(1).to('cuda')
    done = torch.from_numpy(done).float().unsqueeze(1).to('cuda')

    # Update critic

    # Predict next actions with target policy network
    next_action = target_policy_net(next_state)

    # Predict Z distribution with target value network
    target_value = target_value_net.get_probs(next_state, next_action.detach())
    target_z_atoms = value_net.z_atoms

    # Batch of z-atoms
    target_Z_atoms = np.repeat(np.expand_dims(target_z_atoms, axis=0), 256,
                               axis=0)  # [batch_size x n_atoms]
    # Value of terminal states is 0 by definition
    target_Z_atoms[done.cpu().int().numpy(), :] = 0.0

    # Apply bellman update to each atom (expected value)
    reward = reward.cpu().float().numpy()
    #print("Bellman shapes: ", reward.shape, target_Z_atoms.shape, np.expand_dims(gamma, axis=1).shape)
    target_Z_atoms = reward + (target_Z_atoms * gamma)
    #print("after bellman: ", target_Z_atoms.shape)

    #print("Shapes: ", torch.from_numpy(target_Z_atoms).float().shape,
    #      target_value.float().shape, torch.from_numpy(value_net.z_atoms).float().shape)
    target_z_projected = _l2_project(torch.from_numpy(target_Z_atoms).cpu().float(),
                                     target_value.cpu().float(),
                                     torch.from_numpy(value_net.z_atoms).cpu().float())

    #expected_value = reward + (1.0 - done) * gamma * target_value
    #expected_value = torch.clamp(expected_value, min_value, max_value)

    #value = value_net(state, action)
    critic_value = value_net(state, action)
    critic_value = critic_value.to('cuda')
    value_loss = value_criterion(critic_value,
                                 torch.autograd.Variable(target_z_projected, requires_grad=False).cuda())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Update actor

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


if __name__ == "__main__":
    env = NormalizedActions(gym.make("Pendulum-v0"))
    ou_noise = OUNoise(env.action_space)
    device = 'cuda:0'

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    value_lr = 1e-4
    policy_lr = 1e-4

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    value_criterion = nn.BCEWithLogitsLoss()

    replay_buffer_size = 1_000_000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    max_frames = 100_000
    max_steps = 1000
    frame_idx = 0
    rewards = []
    batch_size = 256

    while frame_idx < max_frames:
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy_net.get_action(state)
            action = ou_noise.get_action(action, step)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                ddpg_update(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % max(1000, max_steps+1) == 0:
                print(f"Frame {frame_idx}")
                plot(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)

    plt.savefig('reward_d4pg.png')
