from typing import Protocol, Any

import numpy.typing as npt


class EnvProtocol(Protocol):
    def __init__(self, env_name: str, seed: int) -> None:
        ...

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool, bool, dict[str, Any]]:
        ...

    def reset(self) -> tuple[npt.ArrayLike, dict[str, Any]]:
        ...

    def sample_action(self) -> npt.ArrayLike:
        ...

    def render(self) -> npt.ArrayLike:
        ...

    @property
    def observation_space(self) -> npt.ArrayLike:
        ...

    @property
    def action_space(self) -> npt.ArrayLike:
        ...

    @property
    def env_family(self) -> str:
        ...

