from dataclasses import dataclass, field
import os
import pickle

import numpy as np
import numpy.typing as npt
import torch as t

from oprl.buffers.protocols import ReplayBufferProtocol


@dataclass
class EpisodicReplayBuffer(ReplayBufferProtocol):
    buffer_size_transitions: int
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    max_episode_lenth: int = 1000
    episodes_counter: int = 1
    device: str = "cpu"

    _tensors: dict[str, t.Tensor] = field(init=False)
    _max_episodes: int = field(init=False)
    _ep_pointer: int = 0
    _number_transitions = 0
    _created: bool = False

    def create(self) -> "EpisodicReplayBuffer":
        self._max_episodes = self.buffer_size_transitions // self.max_episode_lenth
        self._tensors = {
            "actions": t.empty(
                (self._max_episodes, self.max_episode_lenth, self.action_dim),
                dtype=t.float32,
                device=self.device,
            ),
            "rewards": t.empty(
                (self._max_episodes, self.max_episode_lenth, 1),
                dtype=t.float32,
                device=self.device
            ),
            "dones": t.empty(
                (self._max_episodes, self.max_episode_lenth, 1),
                dtype=t.float32,
                device=self.device
            ),
            "states": t.empty(
                (self._max_episodes, self.max_episode_lenth + 1, self.state_dim),
                dtype=t.float32,
                device=self.device,
            ),
        }
        self.ep_lens = [0] * self._max_episodes
        self._created = True
        return self

    def check_created(self) -> None:
        if not self._created:
            raise RuntimeError("Replay buffer has to be created with `.create()`.")

    @property
    def states(self) -> t.Tensor:
        self._check_if_created()
        return self._tensors["states"]

    @property
    def actions(self) -> t.Tensor:
        self._check_if_created()
        return self._tensors["actions"]

    @property
    def rewards(self) -> t.Tensor:
        self._check_if_created()
        return self._tensors["rewards"]

    @property
    def dones(self) -> t.Tensor:
        self._check_if_created()
        return self._tensors["dones"]

    def add_transition(self, state: npt.ArrayLike, action: npt.ArrayLike, reward: float, done: bool, episode_done: bool | None = None):
        self.states[self._ep_pointer, self.ep_lens[self._ep_pointer]].copy_(
            t.from_numpy(state)
        )
        self.actions[self._ep_pointer, self.ep_lens[self._ep_pointer]].copy_(
            t.from_numpy(action)
        )
        self.rewards[self._ep_pointer, self.ep_lens[self._ep_pointer]] = float(reward)
        self.dones[self._ep_pointer, self.ep_lens[self._ep_pointer]] = float(done)
        self.ep_lens[self._ep_pointer] += 1
        self._number_transitions = min(self._number_transitions + 1, self.buffer_size_transitions)
        # TODO: Switch to the episodic append and remove condition below
        if episode_done:
            self._inc_episode()

    def _inc_episode(self):
        self._ep_pointer = (self._ep_pointer + 1) % self._max_episodes
        self.episodes_counter = min(self.episodes_counter + 1, self._max_episodes)
        self._number_transitions -= self.ep_lens[self._ep_pointer]
        self.ep_lens[self._ep_pointer] = 0

    def add_episode(self, episode: list):
        for s, a, r, d, _ in episode:
            self.add_transition(s, a, r, d, episode_done=d)
        self._inc_episode()

    def _inds_to_episodic(self, inds):
        start_inds = np.cumsum([0] + self.ep_lens[: self.episodes_counter - 1])
        end_inds = start_inds + np.array(self.ep_lens[: self.episodes_counter])
        ep_inds = np.argmin(
            inds.reshape(-1, 1) >= np.tile(end_inds, (len(inds), 1)), axis=1
        )
        step_inds = inds - start_inds[ep_inds]

        return ep_inds, step_inds

    def sample(self, batch_size):
        inds = np.random.randint(low=0, high=self._number_transitions, size=batch_size)
        ep_inds, step_inds = self._inds_to_episodic(inds)

        return (
            self.states[ep_inds, step_inds],
            self.actions[ep_inds, step_inds],
            self.rewards[ep_inds, step_inds],
            self.dones[ep_inds, step_inds],
            self.states[ep_inds, step_inds + 1],
        )

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        data = {
            "states": self.states.cpu(),
            "actions": self.actions.cpu(),
            "rewards": self.rewards.cpu(),
            "dones": self.dones.cpu(),
            "ep_lens": self.ep_lens,
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Replay buffer saved to {path}")
        except Exception as e:
            print(f"Failed to save replay buffer: {e}")

    @property
    def last_episode_length(self):
        return self.ep_lens[self._ep_pointer]

    def __len__(self) -> int:
        return self._number_transitions
