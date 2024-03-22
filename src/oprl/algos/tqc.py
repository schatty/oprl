import copy

import numpy as np
import numpy.typing as npt
import torch as t
from torch.distributions import Distribution, Normal
from torch.nn import Linear, Module
from torch.nn.functional import logsigmoid, relu

from oprl.utils.logger import StdLogger

LOG_STD_MIN_MAX = (-20, 2)


def quantile_huber_loss_f(quantiles, samples, device):
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


class Critic(Module):
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int, n_nets: int):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [512, 512, 512], n_quantiles)
            self.add_module(f"qf{i}", net)
            self.nets.append(net)

    def forward(self, state: t.Tensor, action: t.Tensor) -> t.Tensor:
        sa = t.cat((state, action), dim=1)
        quantiles = t.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class Actor(Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.action_dim = action_dim
        self.net = Mlp(state_dim, (256, 256), 2 * action_dim)

        self.device = device

    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = t.exp(log_std)
            tanh_normal = TanhNormal(mean, std, self.device)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = t.tanh(mean)
            log_prob = None
        return action, log_prob


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std, device):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(
            t.zeros_like(self.normal_mean, device=device),
            t.ones_like(self.normal_std, device=device),
        )
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return t.tanh(pretanh), pretanh


class Mlp(Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f"fc{i}", fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output


class TQC:
    def __init__(
        self,
        state_shape,
        action_shape,
        discount: float = 0.99,
        tau: float = 0.005,
        top_quantiles_to_drop: int = 2,
        n_quantiles: int = 25,
        n_nets: int = 5,
        log_every: int = 5000,
        device: str = "cpu",
        seed: int = 0,
        logger=StdLogger(),
    ):
        np.random.seed(seed)
        t.manual_seed(seed)

        self.actor = Actor(state_shape[0], action_shape[0], device).to(device)
        self.critic = Critic(state_shape[0], action_shape[0], n_quantiles, n_nets).to(
            device
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = t.tensor(np.log(0.2), requires_grad=True, device=device)

        # TODO: check hyperparams
        self.actor_optimizer = t.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = t.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.alpha_optimizer = t.optim.Adam([self.log_alpha], lr=3e-4)

        self._discount = discount
        self._tau = tau
        self._top_quantiles_to_drop = top_quantiles_to_drop
        self._target_entropy = -np.prod(action_shape[0]).item()
        self._quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        self._device = device
        self._total_it = 0
        self._log_every = log_every
        self._logger = logger

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
                :, : self._quantiles_total - self._top_quantiles_to_drop
            ]

            # compute target
            target = reward + (1 - done) * self._discount * (
                sorted_z_part - alpha * next_log_pi
            )

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target, self._device)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self._tau * param.data + (1 - self._tau) * target_param.data
            )

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self._target_entropy).detach().mean()
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

        if self._total_it % self._log_every == 0:
            self._logger.log_scalar(
                "algo/critic_loss", critic_loss.item(), self._total_it
            )
            self._logger.log_scalar(
                "algo/actor_loss", actor_loss.item(), self._total_it
            )
            self._logger.log_scalar(
                "algo/alpha_loss", alpha_loss.item(), self._total_it
            )

        self._total_it += 1

    def explore(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = t.tensor(state).to(self._device)[None, :]
        action, _ = self.actor(state)
        action = action[0].cpu().detach().numpy()
        return action

    def exploit(self, state: npt.ArrayLike) -> npt.ArrayLike:
        return self.explore(state)
