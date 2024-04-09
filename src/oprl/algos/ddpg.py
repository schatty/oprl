from copy import deepcopy
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import torch as t
from torch import nn

from oprl.algos.nn import Critic, DeterministicPolicy
from oprl.algos.utils import disable_gradient
from oprl.utils.logger import Logger, StdLogger


class DDPG:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1,
        expl_noise: float = 0.1,
        discount: float = 0.99,
        tau: float = 5e-3,
        batch_size: int = 256,
        device: str = "cpu",
        logger: Logger = StdLogger(),
    ):
        self._expl_noise = expl_noise
        self._action_dim = action_dim
        self._discount = discount
        self._tau = tau
        self._batch_size = batch_size
        self._max_action = max_action
        self._device = device
        self._logger = logger

        self.actor = DeterministicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(device)
        self.actor_target = deepcopy(self.actor)
        disable_gradient(self.actor_target)
        self.optim_actor = t.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        disable_gradient(self.critic_target)
        self.optim_critic = t.optim.Adam(self.critic.parameters(), lr=3e-4)

    def update(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor,
    ):
        self._update_critic(state, action, reward, done, next_state)
        self._update_actor(state)

        # Update the frozen target models
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self._tau * param.data + (1 - self._tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self._tau * param.data + (1 - self._tau) * target_param.data
            )

    def _update_critic(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor,
    ) -> None:
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1.0 - done) * self._discount * target_Q.detach()
        current_Q = self.critic(state, action)

        critic_loss = (current_Q - target_Q).pow(2).mean()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

    def _update_actor(self, state: t.Tensor) -> None:
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

    def exploit(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state, device=self._device).unsqueeze_(0)
        with t.no_grad():
            action = self.actor(state).cpu()
        return action.numpy().flatten()

    # TODO: remove explore from algo to agent completely
    def explore(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state, device=self._device).unsqueeze_(0)

        with t.no_grad():
            noise = (
                t.randn(self._action_dim) * self._max_action * self._expl_noise
            ).to(self._device)
            action = self.actor(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self._max_action, self._max_action)

    def get_policy_state_dict(self) -> Dict[str, Any]:
        return self.actor.state_dict()

    @property
    def logger(self) -> Logger:
        return self._logger
