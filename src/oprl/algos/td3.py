from copy import deepcopy

import numpy as np
import numpy.typing as npt
import torch as t
from torch import nn
from torch.optim import Adam

from oprl.algos.nn import DeterministicPolicy, DoubleCritic
from oprl.algos.utils import disable_gradient, soft_update
from oprl.utils.logger import Logger, StdLogger


class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        batch_size: int = 256,
        policy_noise: float = 0.2,
        expl_noise: float = 0.1,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        discount: float = 0.99,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        max_action: float = 1.0,
        tau: float = 5e-3,
        log_every: int = 5000,
        device="cpu",
        logger: Logger = StdLogger(),
    ):
        self._aciton_dim = action_dim
        self._expl_noise = expl_noise
        self._batch_size = batch_size
        self._discount = discount
        self._tau = tau
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_freq = policy_freq
        self._max_action = max_action
        self._device = device
        self._logger = logger

        self._log_every = log_every
        self._update_step = 0

        self.actor = DeterministicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(self._device)
        self.actor_target = deepcopy(self.actor).to(self._device).eval()
        disable_gradient(self.actor_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = DoubleCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(self._device)
        self.critic_target = deepcopy(self.critic).to(self._device).eval()
        disable_gradient(self.critic_target)

        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

    def exploit(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state, device=self._device).unsqueeze_(0)
        with t.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def explore(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state, device=self._device).unsqueeze_(0)
        noise = (t.randn(self._aciton_dim) * self._max_action * self._expl_noise).to(
            self._device
        )

        with t.no_grad():
            action = self.actor(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self._max_action, self._max_action)

    def update(self, state: t.Tensor, action, reward, done, next_state) -> None:
        self._update_critic(state, action, reward, done, next_state)

        if self._update_step % self._policy_freq == 0:
            self._update_actor(state)
            soft_update(self.critic_target, self.critic, self._tau)
            soft_update(self.actor_target, self.actor, self._tau)

        self._update_step += 1

    def _update_critic(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor,
    ) -> None:
        q1, q2 = self.critic(state, action)

        with t.no_grad():
            noise = (t.randn_like(action) * self._policy_noise).clamp(
                -self._noise_clip, self._noise_clip
            )

            next_actions = self.actor_target(next_state) + noise
            next_actions = next_actions.clamp(-self._max_action, self._max_action)

            q1_next, q2_next = self.critic_target(next_state, next_actions)
            q_next = t.min(q1_next, q2_next)

        q_target = reward + (1.0 - done) * self._discount * q_next

        td_error1 = (q1 - q_target).pow(2).mean()
        td_error2 = (q2 - q_target).pow(2).mean()
        loss_critic = td_error1 + td_error2

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self._update_step % self._log_every == 0:
            self._logger.log_scalar(
                "algo/q1", q1.detach().mean().cpu(), self._update_step
            )
            self._logger.log_scalar(
                "algo/q_target", q_target.mean().cpu(), self._update_step
            )
            self._logger.log_scalar(
                "algo/abs_q_err",
                (q1 - q_target).detach().mean().cpu(),
                self._update_step,
            )
            self._logger.log_scalar(
                "algo/critic_loss", loss_critic.item(), self._update_step
            )

    def _update_actor(self, state: t.Tensor) -> None:
        actions = self.actor(state)
        qs1 = self.critic.Q1(state, actions)
        loss_actor = -qs1.mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        if self._update_step % self._log_every == 0:
            self._logger.log_scalar(
                "algo/loss_actor", loss_actor.item(), self._update_step
            )
