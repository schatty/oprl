import copy
from dataclasses import dataclass, field

import numpy as np
import torch as t
import torch.nn as nn

from oprl.algos.protocols import PolicyProtocol
from oprl.algos.base_algorithm import OffPolicyAlgorithm
from oprl.algos.nn_models import MLP, GaussianActor
from oprl.logging import LoggerProtocol


def quantile_huber_loss_f(
    quantiles: t.Tensor, samples: t.Tensor, device: str
) -> t.Tensor:
    """
    quantiles: [batch, n_nets, n_quantiles].
    samples: [batch, n_nets * n_quantiles - top_quantiles_to_drop].
    """
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = t.abs(pairwise_delta)
    huber_loss = t.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )

    n_quantiles = quantiles.shape[2]
    tau = (
        t.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    )
    loss = (
        t.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss


class QuantileQritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int, n_nets: int) -> None:
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = MLP(
                state_dim + action_dim,
                n_quantiles,
                (512, 512, 512),
                hidden_activation=nn.ReLU(),
            )
            self.add_module(f"qf{i}", net)
            self.nets.append(net)

    def forward(self, state: t.Tensor, action: t.Tensor) -> t.Tensor:
        sa = t.cat((state, action), dim=1)
        quantiles = t.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


@dataclass
class TQC(OffPolicyAlgorithm):
    logger: LoggerProtocol
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    tau: float = 0.005
    top_quantiles_to_drop: int = 2
    n_quantiles: int = 25
    n_nets: int = 5
    log_every: int = 5000
    device: str = "cpu"

    actor: PolicyProtocol = field(init=False)
    actor_target: PolicyProtocol = field(init=False)
    actor_optimizer: t.optim.Optimizer = field(init=False)
    critic: QuantileQritic = field(init=False)
    critic_target: QuantileQritic = field(init=False)
    critic_optimizer: t.optim.Optimizer = field(init=False)
    target_entropy: float = field(init=False)
    alpha_optimizer: t.optim.Optimizer = field(init=False)
    quantiles_total: int = field(init=False)
    update_step: int = 0
    _created: bool = False

    def create(self) -> "TQC":
        self.target_entropy = -np.prod(self.action_dim).item()
        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(),
            device=self.device,
        ).to(self.device)
        self.critic = QuantileQritic(
            self.state_dim,
            self.action_dim,
            self.n_quantiles,
            self.n_nets
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = t.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets

        # TODO: check hyperparams
        self.actor_optimizer = t.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = t.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.alpha_optimizer = t.optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self._created = True
        return self

    def update(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor,
    ):
        batch_size = state.shape[0]

        alpha = t.exp(self.log_alpha)

        # --- Q loss ---
        with t.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(
                next_state, new_next_action
            )  # batch x nets x quantiles
            sorted_z, _ = t.sort(next_z.reshape(batch_size, -1))
            sorted_z_part = sorted_z[
                :, : self.quantiles_total - self.top_quantiles_to_drop
            ]

            # compute target
            target = reward + (1 - done) * self.gamma * (
                sorted_z_part - alpha * next_log_pi
            )

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, self.device)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (
            alpha * log_pi
            - self.critic(state, new_action).mean(2).mean(1, keepdim=True)
        ).mean()

        # --- Update ---

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self.update_step % self.log_every == 0:
            self.logger.log_scalars(
                {
                    "algo/critic_loss": critic_loss.item(),
                    "algo/actor_loss": actor_loss.item(),
                    "algo/alpha_loss": alpha_loss.item(),
                },
                self.update_step,
            )

        self.update_step += 1
