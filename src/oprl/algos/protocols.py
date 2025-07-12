from typing import Protocol

import numpy.typing as npt

import torch as t


class PolicyProtocol(Protocol):
    def explore(self, state: npt.ArrayLike) -> npt.NDArray: ...

    def exploit(self, state: npt.ArrayLike) -> npt.NDArray: ...


class AlgorithmProtocol(Protocol):
    actor: PolicyProtocol
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


