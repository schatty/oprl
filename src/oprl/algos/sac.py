from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


from oprl.algos.nn import DoubleCritic, MLP
from oprl.algos.utils import Clamp, initialize_weight, soft_update, disable_gradient


def calculate_gaussian_log_prob(log_stds, noises):
    # NOTE: We only use multivariate gaussian distribution with diagonal
    # covariance matrix,  which can be viewed as simultaneous distribution of
    # gaussian distributions, p_i(u). So, log_probs = \sum_i log p_i(u).
    return (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) \
        - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)


def calculate_log_pi(log_stds, noises, us):
    # NOTE: Because we calculate actions = tanh(us), we need to correct
    # log_probs. Because tanh(u)' = 1 - tanh(u)**2, we need to substract
    # \sum_i log(1 - tanh(u)**2) from gaussian_log_probs. For numerical
    # stabilities, we use the deformation as below.
    # log(1 - tanh(u)**2)
    # = 2 * log(2 / (exp(u) + exp(-u)))
    # = 2 * (log(2) - log(exp(u) * (1 + exp(-2*u))))
    # = 2 * (log(2) - u - softplus(-2*u))
    return calculate_gaussian_log_prob(log_stds, noises) - (
        2 * (math.log(2) - us - F.softplus(-2 * us))
    ).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    x = means + noises * stds
    actions = torch.tanh(x)
    return actions, calculate_log_pi(log_stds, noises, x)


class GaussianPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_shape[0],
            output_dim=hidden_units[-1],
            hidden_units=hidden_units[:-1],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation
        )#.apply(initialize_weight)

        self.mean = nn.Linear(
            hidden_units[-1], action_shape[0]
        ).apply(initialize_weight)

        self.log_std = nn.Sequential(
            nn.Linear(hidden_units[-1], action_shape[0]),
            Clamp()
        ).apply(initialize_weight)

    def forward(self, states):
        return torch.tanh(self.mean(self.mlp(states)))

    def sample(self, states):
        x = self.mlp(states)
        return reparameterize(self.mean(x), self.log_std(x))


class SAC:

    def __init__(self, state_shape, action_shape, device, seed, batch_size=256, tune_alpha=False,
                 gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=1e-3, alpha_init=0.2,
                 target_update_coef=5e-3, log_every=5000, logger=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.update_step = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dtype = torch.uint8 if len(state_shape) == 3 else torch.float
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tune_alpha = tune_alpha
        self.discount = gamma
        self.log_every = log_every

        self.logger = logger

        self.actor = GaussianPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)

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

        self.alpha = alpha_init
        if self.tune_alpha:
            self.log_alpha = torch.tensor(
                np.log(self.alpha), device=device, requires_grad=True)
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            self.target_entropy = -float(action_shape[0])

        self.target_update_coef = target_update_coef

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self, batch):
        states, actions, rewards, dones, next_states = batch  
        self.update_step += 1
        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)

        soft_update(self.critic_target, self.critic, self.target_update_coef)

    def update_critic(self, states, actions, rewards, dones, next_states):
        q1, q2 = self.critic(states, actions)
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_pis

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
        actions, log_pi = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = self.alpha * log_pi.mean() - torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        if self.tune_alpha:
            loss_alpha = -self.log_alpha * (
                self.target_entropy + log_pi.detach_().mean()
            )

            self.optim_alpha.zero_grad()
            loss_alpha.backward()
            self.optim_alpha.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()

        if self.update_step % self.log_every == 0:
            if self.tune_alpha:
                self.logger.log_scalar("algo/loss_alpha", loss_alpha.item(), self.update_step)
            self.logger.log_scalar("algo/loss_actor", loss_actor.item(),  self.update_step)
            self.logger.log_scalar("algo/alpha", self.alpha,  self.update_step)
            self.logger.log_scalar("algo/log_pi", log_pi.cpu().mean(), self.update_step)

    def exploit(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

