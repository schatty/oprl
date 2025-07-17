from abc import ABC
from typing import Any

from oprl.algos.protocols import  AlgorithmProtocol


class OffPolicyAlgorithm(ABC, AlgorithmProtocol):
    def check_created(self) -> None:
        if not self._created:
            raise RuntimeError(
                f"Algorithm {type(self).__name__} has not been created with `create()`."
        )

    def get_policy_state_dict(self) -> dict[str, Any]:
        return self.actor.state_dict()
