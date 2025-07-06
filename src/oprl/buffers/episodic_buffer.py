import os
import pickle
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch as t


class ReplayBufferProtocol(Protocol):
    def add_transition(self, state, action, reward, done, episode_done=None): ...

    def add_episode(self, episode): ...

    def sample(self, batch_size) -> tuple[
        t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor
    ]: ...

    def save(self, path: str) -> None: ...

    @property
    def last_episode_length(self) -> int: ...


class EpisodicReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        device: str,
        gamma: float,
        max_episode_len: int = 1000,
        dtype=t.float,
    ):
        self.buffer_size = buffer_size
        self.max_episodes = buffer_size // max_episode_len
        self.max_episode_len = max_episode_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

        self.ep_pointer = 0
        self.cur_episodes = 1
        self.cur_size = 0

        self.actions = t.empty(
            (self.max_episodes, max_episode_len, action_dim),
            dtype=dtype,
            device=device,
        )
        self.rewards = t.empty(
            (self.max_episodes, max_episode_len, 1), dtype=dtype, device=device
        )
        self.dones = t.empty(
            (self.max_episodes, max_episode_len, 1), dtype=dtype, device=device
        )
        self.states = t.empty(
            (self.max_episodes, max_episode_len + 1, state_dim),
            dtype=dtype,
            device=device,
        )
        self.ep_lens = [0] * self.max_episodes

        self.actions_for_std = t.empty(
            (100, action_dim), dtype=dtype, device=device
        )
        self.actions_for_std_cnt = 0

    def add_transition(self, state: npt.ArrayLike, action: npt.ArrayLike, reward: float, done: bool, episode_done: bool | None = None):
        self.states[self.ep_pointer, self.ep_lens[self.ep_pointer]].copy_(
            t.from_numpy(state)
        )
        self.actions[self.ep_pointer, self.ep_lens[self.ep_pointer]].copy_(
            t.from_numpy(action)
        )
        self.rewards[self.ep_pointer, self.ep_lens[self.ep_pointer]] = float(reward)
        self.dones[self.ep_pointer, self.ep_lens[self.ep_pointer]] = float(done)

        self.actions_for_std[self.actions_for_std_cnt % 100].copy_(
            t.from_numpy(action)
        )
        self.actions_for_std_cnt += 1

        self.ep_lens[self.ep_pointer] += 1
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
        if episode_done:
            self._inc_episode()

    def _inc_episode(self):
        self.ep_pointer = (self.ep_pointer + 1) % self.max_episodes
        self.cur_episodes = min(self.cur_episodes + 1, self.max_episodes)
        self.cur_size -= self.ep_lens[self.ep_pointer]
        self.ep_lens[self.ep_pointer] = 0

    def add_episode(self, episode: list):
        for s, a, r, d, _ in episode:
            self.add_transition(s, a, r, d, episode_done=d)
            if d:
                break
        else:
            self._inc_episode()

    def _inds_to_episodic(self, inds):
        start_inds = np.cumsum([0] + self.ep_lens[: self.cur_episodes - 1])
        end_inds = start_inds + np.array(self.ep_lens[: self.cur_episodes])
        ep_inds = np.argmin(
            inds.reshape(-1, 1) >= np.tile(end_inds, (len(inds), 1)), axis=1
        )
        step_inds = inds - start_inds[ep_inds]

        return ep_inds, step_inds

    def sample(self, batch_size):
        inds = np.random.randint(low=0, high=self.cur_size, size=batch_size)
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

    def __len__(self) -> int:
        return self.cur_size

    # @property
    # def num_episodes(self):
    #     return self.cur_episodes

    @property
    def last_episode_length(self):
        return self.ep_lens[self.ep_pointer]
