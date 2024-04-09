from copy import deepcopy

import numpy as np
import numpy.typing as npt
import torch as t
from torch import nn
from torch.optim import Adam

from oprl.algos.nn import DoubleCritic, GaussianActor
from oprl.algos.utils import disable_gradient, soft_update
from oprl.utils.logger import Logger, StdLogger


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
        log_every: int = 5000,
        logger: Logger = StdLogger(),
    ):
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
        ).to(device)

        self.critic = DoubleCritic(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(device)

        self.critic_target = deepcopy(self.critic).to(device).eval()
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
