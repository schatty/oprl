import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import OUNoise
from utils.reward_plot import plot_rewards
from utils.logger import Logger

from .networks import PolicyNetwork, ValueNetwork
from .l2_projection import _l2_project


class LearnerD4PG(object):
    """Policy and value network update routine. """

    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        hidden_dim = config['dense_size']
        state_dim = config['state_dims']
        action_dim = config['action_dims']
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        v_min = config['v_min']
        v_max = config['v_max']
        num_atoms = config['num_atoms']
        self.device = config['device']
        self.max_steps = config['max_ep_length']
        self.num_train_steps = config['num_steps_train']
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.gamma = config['discount_rate']
        self.log_dir = log_dir
        self.prioritized_replay = config['replay_memory_prioritized']
        self.learner_w_queue = learner_w_queue

        self.logger = Logger(f"{log_dir}/learner")

        # Noise process
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])

        # Value and policy nets
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim, v_min, v_max, num_atoms, device=self.device)
        self.policy_net = policy_net #PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim, v_min, v_max, num_atoms, device=self.device)
        self.target_policy_net = target_policy_net

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.BCELoss(reduction='none')

    def ddpg_update(self, batch, replay_priority_queue, update_step, min_value=-np.inf, max_value=np.inf):
        update_time = time.time()

        state, action, reward, next_state, done, gamma, weights, inds = batch

        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)
        weights = np.asarray(weights)
        inds = np.asarray(inds).flatten()

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

        critic_value = self.value_net.get_probs(state, action)#self.value_net(state, action)

        critic_value = critic_value.to(self.device)

        value_loss = self.value_criterion(critic_value,
                                     torch.autograd.Variable(target_z_projected, requires_grad=False).cuda())

        value_loss = value_loss.mean(axis=1)

        # Update priorities in buffer
        td_error = value_loss.cpu().detach().numpy().flatten()

        priority_epsilon = 1e-4
        if self.prioritized_replay:
            weights_update = np.abs(td_error) + priority_epsilon
            replay_priority_queue.put((inds, weights_update))
            value_loss = value_loss * torch.tensor(weights).cuda().float()

        value_loss = value_loss.mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------- Update actor -----------

        policy_loss = self.value_net.get_probs(state, self.policy_net(state))
        policy_loss = policy_loss * torch.tensor(self.value_net.z_atoms).float().cuda()
        policy_loss = torch.sum(policy_loss, dim=1)
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

        # Send updated learner to the queue
        if not self.learner_w_queue.full():
            params = [p.data.cpu().detach().numpy() for p in self.policy_net.parameters()]
            self.learner_w_queue.put(params)

        # Logging
        step = update_step.value
        self.logger.scalar_summary("learner/policy_loss", policy_loss.item(), step)
        self.logger.scalar_summary("learner/value_loss", value_loss.item(), step)
        self.logger.scalar_summary("learner/learner_update_timing", time.time() - update_time, step)

    def run(self, training_on, batch_queue, replay_priority_queue, update_step):
        while update_step.value < self.num_train_steps:
            if batch_queue.empty():
                continue

            batch = batch_queue.get()
            self.ddpg_update(batch, replay_priority_queue, update_step)
            update_step.value += 1

            if update_step.value % 50 == 0:
                print("Training step ", update_step.value)

        training_on.value = 0
        plot_rewards(self.log_dir)
        print("Exit learner.")
