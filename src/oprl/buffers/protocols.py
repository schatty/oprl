from typing import Protocol

import torch as t


class ReplayBufferProtocol(Protocol):
    def create(self) -> "ReplayBufferProtocol": ...
    def add_transition(self, state, action, reward, done, episode_done=None): ...

    def add_episode(self, episode): ...

    def sample(self, batch_size) -> tuple[
        t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor
    ]: ...

    def save(self, path: str) -> None: ...

    @property
    def last_episode_length(self) -> int: ...

