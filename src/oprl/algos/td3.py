from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


from oprl.algos.nn import DoubleCritic, MLP
from oprl.algos.utils import initialize_weight, soft_update, disable_gradient


class DeterministicPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).apply(initialize_weight)

    def forward(self, states):
        return torch.tanh(self.mlp(states))


class TD3:

    def __init__(self, state_shape, action_shape, device, seed, batch_size=256, policy_noise=0.2,
                 expl_noise=0.1, noise_clip=0.5, policy_freq=2, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
                 max_action=1.0, target_update_coef=5e-3, log_every=5000, logger=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.update_step = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dtype = torch.uint8 if len(state_shape) == 3 else torch.float
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.discount = gamma
        self.log_every = log_every

        self.logger = logger

        self.actor = DeterministicPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)

        self.actor_target = deepcopy(self.actor).to(self.device).eval()

        self.critic = DoubleCritic(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)

        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.target_update_coef = target_update_coef

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)

        with torch.no_grad():
            noise = (torch.randn(self.action_shape) * self.max_action * self.expl_noise).to(self.device)
            action = self.actor(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self.max_action, self.max_action)

    def update(self, batch):
        self.update_step += 1

        state, action, reward, done, next_state = batch
        self.update_critic(state, action, reward, done, next_state)

        if self.update_step % self.policy_freq == 0:
            self.update_actor(state)
            soft_update(self.critic_target, self.critic, self.target_update_coef)
            soft_update(self.actor_target, self.actor, self.target_update_coef)

    def update_critic(self, states, actions, rewards, dones, next_states):
        q1, q2 = self.critic(states, actions)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

        q_target = rewards + (1.0 - dones) * self.discount * q_next

        td_error1 = (q1 - q_target).pow(2).mean()
        td_error2 = (q2 - q_target).pow(2).mean()
        loss_critic = td_error1 + td_error2

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self.update_step % self.log_every == 0:
            self.logger.log_scalar("algo/q1", q1.detach().mean().cpu(), self.update_step)
            self.logger.log_scalar("algo/q_target", q_target.mean().cpu(), self.update_step)
            self.logger.log_scalar("algo/abs_q_err", (q1 - q_target).detach().mean().cpu(), self.update_step)
            self.logger.log_scalar("algo/critic_loss", loss_critic.item(), self.update_step)
            self.logger.log_scalar("algo/q1_grad_norm", self.critic.q1.get_layer_norm(), self.update_step)
            self.logger.log_scalar("algo/actor_grad_norm", self.actor.mlp.get_layer_norm(), self.update_step)
            
    def update_actor(self, states):
        actions = self.actor(states)
        qs1 = self.critic.Q1(states, actions)
        loss_actor = -qs1.mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        if self.update_step % self.log_every == 0:
            self.logger.log_scalar("algo/loss_actor", loss_actor.item(), self.update_step)

    def exploit(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]
