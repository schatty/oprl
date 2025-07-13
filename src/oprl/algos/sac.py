from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch as t
from torch import nn
from torch.optim import Adam

from oprl.algos.protocols import PolicyProtocol
from oprl.algos.base_algorithm import OffPolicyAlgorithm
from oprl.algos.nn_models import DoubleCritic, GaussianActor
from oprl.algos.nn_functions import disable_gradient, soft_update
from oprl.logging import LoggerProtocol


@dataclass
class SAC(OffPolicyAlgorithm):
    logger: LoggerProtocol
    state_dim: int
    action_dim: int
    batch_size: int = 256
    tune_alpha: bool = False
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 1e-3
    alpha_init: float = 0.2
    target_update_coef: float = 5e-3
    device: str = "cpu"
    log_every: int = 5000
 
    actor: PolicyProtocol = field(init=False)
    actor_target: PolicyProtocol = field(init=False)
    optim_actor: t.optim.Optimizer = field(init=False)
    critic: nn.Module = field(init=False)
    critic_target: nn.Module = field(init=False)
    optim_critic: t.optim.Optimizer = field(init=False)
    alpha: float = field(init=False)
    update_step: int = 0

    def create(self) -> "SAC":
        self.actor = GaussianActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
            device=self.device,
        ).to(self.device)

        self.critic = DoubleCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(self.device)

        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=self.lr_critic)

        self.alpha = self.alpha_init
        if self.tune_alpha:
            self.log_alpha = t.tensor(
                np.log(self.alpha), device=self.device, requires_grad=True
            )
            self.optim_alpha = t.optim.Adam([self.log_alpha], lr=self.lr_alpha)
            self.target_entropy = -float(self.action_dim)

        return self

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
        soft_update(self.critic_target, self.critic, self.target_update_coef)
        self.update_step += 1

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
            q_next = t.min(q1_next, q2_next) - self.alpha * log_pis

        q_target = rewards + (1.0 - dones) * self.gamma * q_next

        td_error1 = (q1 - q_target).pow(2).mean()
        td_error2 = (q2 - q_target).pow(2).mean()
        loss_critic = td_error1 + td_error2

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self.update_step % self.log_every == 0:
            self.logger.log_scalars(
                {
                    "algo/q1": q1.detach().mean().cpu(),
                    "algo/q_target": q_target.mean().cpu(),
                    "algo/abs_q_err": (q1 - q_target).detach().mean().cpu(),
                    "algo/critic_loss": loss_critic.item(),
                },
                self.update_step,
            )

    def update_actor(self, state: t.Tensor) -> None:
        actions, log_pi = self.actor(state)
        qs1, qs2 = self.critic(state, actions)
        loss_actor = self.alpha * log_pi.mean() - t.min(qs1, qs2).mean()

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
            with t.no_grad():
                self.alpha = self.log_alpha.exp().item()

        if self.update_step % self.log_every == 0:
            if self.tune_alpha:
                self.logger.log_scalar(
                    "algo/loss_alpha", loss_alpha.item(), self.update_step
                )
            self.logger.log_scalars(
                {
                    "algo/loss_actor": loss_actor.item(),
                    "algo/alpha": self._alpha,
                    "algo/log_pi": log_pi.cpu().mean(),
                },
                self.update_step,
            )
