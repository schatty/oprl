from typing import Protocol

import numpy.typing as npt

import torch as t


class PolicyProtocol(Protocol):
    def explore(self, state: npt.ArrayLike) -> npt.NDArray: ...

    def exploit(self, state: npt.ArrayLike) -> npt.NDArray: ...


class AlgorithmProtocol(Protocol):
    actor: PolicyProtocol

    def create(self) -> "AlgorithmProtocol": ...

    def update(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor
    ) -> None: ...


