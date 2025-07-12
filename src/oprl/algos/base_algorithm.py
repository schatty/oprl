from abc import ABC
from typing import Any

import numpy.typing as npt

from oprl.algos.protocols import  AlgorithmProtocol


class OffPolicyAlgorithm(ABC, AlgorithmProtocol):
    def check_created(self) -> None:
        if not self._created:
            raise RuntimeError(
                f"Algorithm {type(self).__name__} has not been created with `create()`."
        )

    def exploit(self, state: npt.NDArray) -> npt.NDArray:
        return self.actor.exploit(state)

    def explore(self, state: npt.NDArray) -> npt.NDArray:
        return self.actor.explore(state)

    def get_policy_state_dict(self) -> dict[str, Any]:
        return self.actor.state_dict()
