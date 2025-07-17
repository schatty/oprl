from typing import Protocol, runtime_checkable

import torch as t


@runtime_checkable
class ReplayBufferProtocol(Protocol):
    episodes_counter: int
    _created: bool

    def create(self) -> "ReplayBufferProtocol": ...

    def check_created(self) -> None: ...

    def add_transition(self, state, action, reward, done, episode_done=None): ...

    def add_episode(self, episode): ...

    def sample(self, batch_size) -> tuple[
        t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor
    ]: ...

    def __len__(self) -> int: ...

    @property
    def last_episode_length(self) -> int: ...

