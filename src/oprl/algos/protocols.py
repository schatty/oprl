from typing import Protocol, Any

import numpy.typing as npt

import torch as t
import torch.nn as nn

from oprl.logging import LoggerProtocol


class PolicyProtocol(Protocol):
    def explore(self, state: npt.NDArray) -> npt.NDArray: ...

    def exploit(self, state: npt.NDArray) -> npt.NDArray: ...

    def __call__(*args, **kwargs) -> t.Tensor: ...


class AlgorithmProtocol(Protocol):
    actor: PolicyProtocol
    critic: nn.Module
    logger: LoggerProtocol
    _created: bool

    def create(self) -> "AlgorithmProtocol": ...

    def check_created(self) -> None: ...

    def update(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor
    ) -> None: ...

    def get_policy_state_dict(self) -> dict[str, Any]:
        return self.actor.state_dict()

