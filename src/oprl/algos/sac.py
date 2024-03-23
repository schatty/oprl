import math
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from oprl.algos.nn import MLP, DoubleCritic, GaussianActor
from oprl.algos.utils import Clamp, disable_gradient, initialize_weight, soft_update
from oprl.utils.logger import Logger, StdLogger

"""
def calculate_gaussian_log_prob(log_stds: t.Tensor, noises: t.Tensor) -> t.Tensor:
    # NOTE: We only use multivariate gaussian distribution with diagonal
    # covariance matrix,  which can be viewed as simultaneous distribution of
    # gaussian distributions, p_i(u). So, log_probs = \sum_i log p_i(u).
    return (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(
        2 * math.pi
    ) * log_stds.size(-1)


def calculate_log_pi(log_stds: t.Tensor, noises: t.Tensor, us: t.Tensor) -> t.Tensor:
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


def reparameterize(means: t.Tensor, log_stds: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    stds = log_stds.exp()
    noises = t.randn_like(means)
    x = means + noises * stds
    actions = t.tanh(x)
    return actions, calculate_log_pi(log_stds, noises, x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: tuple[int, ...] = (256, 256),
        hidden_activation: t.nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_dim,
            output_dim=hidden_units[-1],
            hidden_units=hidden_units[:-1],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
        )

        self.mean = nn.Linear(hidden_units[-1], action_dim).apply(initialize_weight)

        self.log_std = nn.Sequential(
            nn.Linear(hidden_units[-1], action_dim), Clamp()
        ).apply(initialize_weight)

    def forward(self, states: t.Tensor) -> t.Tensor:
        return t.tanh(self.mean(self.mlp(states)))

    def sample(self, states: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        x = self.mlp(states)
        return reparameterize(self.mean(x), self.log_std(x))
"""


class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        batch_size: int = 256,
        tune_alpha: bool = False,
        gamma: float = 0.99,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 1e-3,
        alpha_init: float = 0.2,
        target_update_coef: float = 5e-3,
        device: str = "cpu",
        seed: int = 0,
        log_every: int = 5000,
        logger: Logger = StdLogger(),
    ):
        np.random.seed(seed)
        t.manual_seed(seed)

        self._update_step = 0
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._batch_size = batch_size
        self._gamma = gamma
        self._tune_alpha = tune_alpha
        self._discount = gamma
        self._target_update_coef = target_update_coef
        self._log_every = log_every

        self.actor = GaussianActor(
            state_dim=self._state_dim,
            action_dim=action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
            device=self._device,
        )

        self.critic = DoubleCritic(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(self._device)

        self.critic_target = deepcopy(self.critic).to(self._device).eval()
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self._alpha = alpha_init
        if self._tune_alpha:
            self.log_alpha = t.tensor(
                np.log(self._alpha), device=device, requires_grad=True
            )
            self.optim_alpha = t.optim.Adam([self.log_alpha], lr=lr_alpha)
            self._target_entropy = -float(action_dim)

        self._logger = logger

    def update(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor,
    ) -> None:
        self.update_critic(state, action, reward, done, next_state)
        self.update_actor(state)
        soft_update(self.critic_target, self.critic, self._target_update_coef)

        self._update_step += 1

    def update_critic(
        self,
        states: t.Tensor,
        actions: t.Tensor,
        rewards: t.Tensor,
        dones: t.Tensor,
        next_states: t.Tensor,
    ) -> None:
        q1, q2 = self.critic(states, actions)
        with t.no_grad():
            next_actions, log_pis = self.actor(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = t.min(q1_next, q2_next) - self._alpha * log_pis

        q_target = rewards + (1.0 - dones) * self._discount * q_next

        td_error1 = (q1 - q_target).pow(2).mean()
        td_error2 = (q2 - q_target).pow(2).mean()
        loss_critic = td_error1 + td_error2

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self._update_step % self._log_every == 0:
            self._logger.log_scalars(
                {
                    "algo/q1": q1.detach().mean().cpu(),
                    "algo/q_target": q_target.mean().cpu(),
                    "algo/abs_q_err": (q1 - q_target).detach().mean().cpu(),
                    "algo/critic_loss": loss_critic.item(),
                },
                self._update_step,
            )

    def update_actor(self, state: t.Tensor) -> None:
        actions, log_pi = self.actor(state)
        qs1, qs2 = self.critic(state, actions)
        loss_actor = self._alpha * log_pi.mean() - t.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        if self._tune_alpha:
            loss_alpha = -self.log_alpha * (
                self._target_entropy + log_pi.detach_().mean()
            )

            self.optim_alpha.zero_grad()
            loss_alpha.backward()
            self.optim_alpha.step()
            with t.no_grad():
                self._alpha = self.log_alpha.exp().item()

        if self._update_step % self._log_every == 0:
            if self._tune_alpha:
                self._logger.log_scalar(
                    "algo/loss_alpha", loss_alpha.item(), self._update_step
                )
            self._logger.log_scalars(
                {
                    "algo/loss_actor": loss_actor.item(),
                    "algo/alpha": self._alpha,
                    "algo/log_pi": log_pi.cpu().mean(),
                },
                self._update_step,
            )

    def explore(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state, device=self._device).unsqueeze_(0)
        with t.no_grad():
            action, _ = self.actor(state)
        return action.cpu().numpy()[0]

    def exploit(self, state: npt.ArrayLike) -> npt.ArrayLike:
        self.actor.eval()
        action = self.explore(state)
        self.actor.train()
        return action
