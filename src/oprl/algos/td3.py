from copy import deepcopy
from dataclasses import dataclass, field

import torch as t
from torch import nn
from torch.optim import Adam

from oprl.algos.protocols import PolicyProtocol
from oprl.algos.base_algorithm import OffPolicyAlgorithm
from oprl.algos.nn_models import DeterministicPolicy, DoubleCritic
from oprl.algos.nn_functions import disable_gradient, soft_update
from oprl.logging import LoggerProtocol


@dataclass
class TD3(OffPolicyAlgorithm):
    logger: LoggerProtocol
    state_dim: int
    action_dim: int
    batch_size: int = 256
    policy_noise: float = 0.2
    expl_noise: float = 0.1
    noise_clip: float = 0.5
    policy_freq: int = 2
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    max_action: float = 1.0
    tau: float = 5e-3
    log_every: int = 5000
    device: str = "cpu"

    actor: PolicyProtocol = field(init=False)
    actor_target: PolicyProtocol = field(init=False)
    optim_actor: t.optim.Optimizer = field(init=False)
    critic: nn.Module = field(init=False)
    critic_target: nn.Module = field(init=False)
    optim_critic: t.optim.Optimizer = field(init=False)
    update_step: int = 0
    _created: bool = False

    def create(self) -> "TD3":
        self.actor = DeterministicPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
            expl_noise=self.expl_noise,
            device=self.device,
        ).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device).eval()
        disable_gradient(self.actor_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic = DoubleCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True),
        ).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        disable_gradient(self.critic_target)

        self.optim_critic = Adam(self.critic.parameters(), lr=self.lr_critic)

        self._created = True
        return self


    def update(
            self,
            state: t.Tensor,
            action: t.Tensor,
            reward: t.Tensor,
            done: t.Tensor,
            next_state: t.Tensor,
        ) -> None:
        self._update_critic(state, action, reward, done, next_state)

        if self.update_step % self.policy_freq == 0:
            self._update_actor(state)
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)
        self.update_step += 1

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
            noise = (t.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_actions = self.actor_target(next_state) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)

            q1_next, q2_next = self.critic_target(next_state, next_actions)
            q_next = t.min(q1_next, q2_next)

        q_target = reward + (1.0 - done) * self.gamma * q_next

        td_error1 = (q1 - q_target).pow(2).mean()
        td_error2 = (q2 - q_target).pow(2).mean()
        loss_critic = td_error1 + td_error2

        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        if self.update_step % self.log_every == 0:
            self.logger.log_scalar(
                "algo/q1", q1.detach().mean().cpu(), self.update_step
            )
            self.logger.log_scalar(
                "algo/q_target", q_target.mean().cpu(), self.update_step
            )
            self.logger.log_scalar(
                "algo/abs_q_err",
                (q1 - q_target).detach().mean().cpu(),
                self.update_step,
            )
            self.logger.log_scalar(
                "algo/critic_loss", loss_critic.item(), self.update_step
            )

    def _update_actor(self, state: t.Tensor) -> None:
        actions = self.actor(state)
        qs1 = self.critic.Q1(state, actions)
        loss_actor = -qs1.mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        if self.update_step % self.log_every == 0:
            self.logger.log_scalar(
                "algo/loss_actor", loss_actor.item(), self.update_step
            )
