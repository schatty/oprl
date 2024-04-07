import numpy as np
import numpy.typing as npt
import torch as t
import torch.nn as nn
from torch.distributions import Distribution, Normal
from torch.nn.functional import logsigmoid

from oprl.algos.utils import initialize_weight

LOG_STD_MIN_MAX = (-20, 2)


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: tuple[int, ...] = (256, 256),
        hidden_activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.q1 = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )

    def forward(self, states: t.Tensor, actions: t.Tensor):
        x = t.cat([states, actions], dim=-1)
        return self.q1(x)

    def Q1(self, states: t.Tensor, actions: t.Tensor) -> t.Tensor:
        x = t.cat([states, actions], dim=-1)
        return self.q1(x)


class DoubleCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: tuple[int, ...] = (256, 256),
        hidden_activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.q1 = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )

        self.q2 = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )

    def forward(self, states: t.Tensor, actions: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        x = t.cat([states, actions], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, states: t.Tensor, actions: t.Tensor) -> t.Tensor:
        x = t.cat([states, actions], dim=-1)
        return self.q1(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: tuple[int, ...] = (64, 64),
        hidden_activation: nn.Module = nn.Tanh(),
        output_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()

        layers = []
        units = input_dim
        for next_units in hidden_units:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        layers.append(nn.Linear(units, output_dim))
        layers.append(output_activation)

        self.nn = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.nn(x)


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: tuple[int, ...] = (256, 256),
        hidden_activation=nn.ReLU(inplace=True),
        max_action: float = 1.0,
        expl_noise: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).apply(initialize_weight)

        self._device = device
        self._action_shape = action_dim
        self._max_action = max_action
        self._expl_noise = expl_noise

    def forward(self, states: t.Tensor) -> t.Tensor:
        return t.tanh(self.mlp(states))

    def exploit(self, state: npt.ArrayLike) -> npt.NDArray:
        state = t.tensor(state).unsqueeze_(0).to(self._device)
        return self.forward(state).cpu().numpy().flatten()

    def explore(self, state: npt.ArrayLike) -> npt.NDArray:
        state = t.tensor(state, device=self._device).unsqueeze_(0)

        with t.no_grad():
            noise = (t.randn(self._action_shape) * self._expl_noise).to(self._device)
            action = self.mlp(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self._max_action, self._max_action)


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, hidden_activation):
        super().__init__()
        self.action_dim = action_dim
        self.net = MLP(
            state_dim, 2 * action_dim, hidden_units, hidden_activation=hidden_activation
        )

    def forward(self, obs: t.Tensor) -> tuple[t.Tensor, t.Tensor | None]:
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

    @property
    def device(self):
        return next(self.parameters()).device


class TanhNormal(Distribution):
    def __init__(self, normal_mean: t.Tensor, normal_std: t.Tensor, device: str):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(
            t.zeros_like(self.normal_mean, device=device),
            t.ones_like(self.normal_std, device=device),
        )
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh: t.Tensor) -> t.Tensor:
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self) -> tuple[t.Tensor, t.Tensor]:
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return t.tanh(pretanh), pretanh
