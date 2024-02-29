import numpy as np
import numpy.typing as npt
import torch as t
import torch.nn as nn

from oprl.algos.utils import initialize_weight


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

        self.nn = nn.Sequential(*layers).apply(initialize_weight)

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

    def forward(self, states: t.Tensor) -> t.Tensor:
        return t.tanh(self.mlp(states))

    def exploit(self, state: npt.ArrayLike) -> npt.NDArray:
        state = t.tensor(state).unsqueeze_(0).to(self._device)
        return self.forward(state).cpu().numpy().flatten()

    def explore(self, state: npt.ArrayLike) -> npt.NDArray:
        state = t.tensor(state, device=self._device).unsqueeze_(0)

        # TODO: Set exploration noise of 0.1 as the property of the agent
        with t.no_grad():
            noise = (t.randn(self._action_shape) * 0.1).to(self._device)
            action = self.mlp(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self._max_action, self._max_action)
