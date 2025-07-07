from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch as t
from torch import nn

from oprl.algos import OffPolicyAlgorithm
from oprl.algos.nn import Critic, DeterministicPolicy
from oprl.algos.utils import disable_gradient
from oprl.logging import LoggerProtocol


@dataclass
class DDPG(OffPolicyAlgorithm):
    state_dim: int
    action_dim: int
    logger: LoggerProtocol
    expl_noise: float = 0.1
    discount: float = 0.99
    tau: float = 5e-3
    batch_size: int = 256
    max_action: float = 1.
    device: str = "cpu"

    def create(self) -> "DDPG":
        self.actor = DeterministicPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(self.device)
        self.actor_target = deepcopy(self.actor)
        disable_gradient(self.actor_target)
        self.optim_actor = t.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)
        disable_gradient(self.critic_target)
        self.optim_critic = t.optim.Adam(self.critic.parameters(), lr=3e-4)
        return self

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
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
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
        target_Q = reward + (1.0 - done) * self.discount * target_Q.detach()
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
        state = t.tensor(state, device=self.device).unsqueeze_(0)
        with t.no_grad():
            action = self.actor(state).cpu()
        return action.numpy().flatten()

    # TODO: remove explore from algo to agent completely
    def explore(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state, device=self.device).unsqueeze_(0)

        with t.no_grad():
            noise = (
                t.randn(self.action_dim) * self.max_action * self.expl_noise
            ).to(self.device)
            action = self.actor(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self.max_action, self.max_action)

    def get_policy_state_dict(self) -> dict[str, Any]:
        return self.actor.state_dict()
