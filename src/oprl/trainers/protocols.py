from typing import Protocol

class TrainerProtocol(Protocol):
    def train(self) -> None: ...

    def evaluate(self) -> dict[str, float]: ...

