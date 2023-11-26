import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import queue

from utils.utils import OUNoise, empty_torch_queue
from utils.logger import Logger

from .networks import ValueNetwork
from .l2_projection import _l2_project


class LearnerD4PG(object):
    """Policy and value network update routine. """

    def __init__(self, config, policy_net, target_policy_net, learner_w_queue, log_dir=''):
        hidden_dim = config['dense_size']
        state_dim = config['state_dim']
        action_dim = config['action_dim']
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        self.v_min = config['v_min']
        self.v_max = config['v_max']
        self.num_atoms = config['num_atoms']
        self.device = config['device']
        self.max_steps = config['max_ep_length']
        self.num_train_steps = config['num_steps_train']
        self.batch_size = config['batch_size']
        self.tau = config['tau']
        self.gamma = config['discount_rate']
        self.n_step_return = config["n_step_returns"]
        self.log_dir = log_dir
        self.prioritized_replay = config['replay_memory_prioritized']
        self.learner_w_queue = learner_w_queue
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.logger = Logger(f"{log_dir}/learner")

        # Noise process
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])

        # Value and policy nets
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim, self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.policy_net = policy_net
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim, self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.target_policy_net = target_policy_net

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.BCELoss(reduction='none')

    def _update_step(self, batch, replay_priority_queue, update_step):
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
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        # ------- Update critic -------

        # Predict next actions with target policy network
        next_action = self.target_policy_net(next_state)

        # Predict Z distribution with target value network
        target_value = self.target_value_net.get_probs(next_state, next_action.detach())

        # Get projected distribution
        target_z_projected = _l2_project(next_distr_v=target_value,
                                         rewards_v=reward,
                                         dones_mask_t=done,
                                         gamma=self.gamma ** self.n_step_return,
                                         n_atoms=self.num_atoms,
                                         v_min=self.v_min,
                                         v_max=self.v_max,
                                         delta_z=self.delta_z)
        target_z_projected = torch.from_numpy(target_z_projected).float().to(self.device)

        critic_value = self.value_net.get_probs(state, action)
        critic_value = critic_value.to(self.device)

        value_loss = self.value_criterion(critic_value, target_z_projected)
        value_loss = value_loss.mean(axis=1)

        # Update priorities in buffer
        td_error = value_loss.cpu().detach().numpy().flatten()
        priority_epsilon = 1e-4
        if self.prioritized_replay:
            weights_update = np.abs(td_error) + priority_epsilon
            replay_priority_queue.put((inds, weights_update))
            value_loss = value_loss * torch.tensor(weights).float().to(self.device)

        # Update step
        value_loss = value_loss.mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------- Update actor -----------

        policy_loss = self.value_net.get_probs(state, self.policy_net(state))
        policy_loss = policy_loss * torch.from_numpy(self.value_net.z_atoms).float().to(self.device)
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
        if update_step.value % 100 == 0:
            try:
                params = [p.data.cpu().detach().numpy() for p in self.policy_net.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass

        # Logging
        step = update_step.value
        self.logger.scalar_summary("learner/policy_loss", policy_loss.item(), step)
        self.logger.scalar_summary("learner/value_loss", value_loss.item(), step)
        self.logger.scalar_summary("learner/learner_update_timing", time.time() - update_time, step)

    def run(self, training_on, batch_queue, replay_priority_queue, update_step):
        torch.set_num_threads(4)
        while update_step.value < self.num_train_steps:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
                continue

            self._update_step(batch, replay_priority_queue, update_step)
            update_step.value += 1

            if update_step.value % 1000 == 0:
                print("Training step ", update_step.value)

        training_on.value = 0

        empty_torch_queue(self.learner_w_queue)
        empty_torch_queue(replay_priority_queue)
        print("Exit learner.")
