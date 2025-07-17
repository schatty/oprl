import numpy.typing as npt
from typing import Any

import gymnasium as gym

from oprl.environment.protocols import EnvProtocol


class Gymnasium(EnvProtocol):
    def __init__(self, env_name: str, seed: int) -> None:
        self._env = gym.make(env_name)
        self._seed = seed

    def step(
        self,
        action: npt.NDArray,
    ) -> tuple[npt.NDArray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype("float32"), float(reward), terminated, bool(truncated), info

    def reset(self) -> tuple[npt.NDArray, dict[str, Any]]:
        obs, info = self._env.reset(seed=self._seed)
        self._env.step(self._env.action_space.sample())
        return obs.astype("float32"), info

    def sample_action(self) -> npt.NDArray:
        return self._env.action_space.sample()

    def render(self) -> npt.NDArray:
        return self._env.render()

    @property
    def observation_space(self) -> npt.NDArray:
        return self._env.observation_space

    @property
    def action_space(self) -> npt.NDArray:
        return self._env.action_space

    @property
    def env_family(self) -> str:
        return "safety_gymnasium"
