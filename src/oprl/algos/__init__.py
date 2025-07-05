from typing import Protocol

import torch as t


class OffPolicyAlgorithm(Protocol):
    def create(self) -> "OffPolicyAlgorithm": ...

    def update(
        self,
        state: t.Tensor,
        action: t.Tensor,
        reward: t.Tensor,
        done: t.Tensor,
        next_state: t.Tensor
    ) -> None: ...

