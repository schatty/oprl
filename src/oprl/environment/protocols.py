from typing import Protocol, Any

import numpy.typing as npt


class EnvProtocol(Protocol):
    def __init__(self, env_name: str, seed: int) -> None:
        ...

    def step(
        self, action: npt.NDArray
    ) -> tuple[npt.NDArray, float, bool, bool, dict[str, Any]]:
        ...

    def reset(self) -> tuple[npt.NDArray, dict[str, Any]]:
        ...

    def sample_action(self) -> npt.NDArray:
        ...

    def render(self) -> npt.NDArray:
        ...

    @property
    def observation_space(self) -> npt.NDArray:
        ...

    @property
    def action_space(self) -> npt.NDArray:
        ...

    @property
    def env_family(self) -> str:
        ...

