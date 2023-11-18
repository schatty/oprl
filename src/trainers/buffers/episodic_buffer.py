import os

import pickle
import numpy as np
import torch


class EpisodicReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, gamma, max_episode_len=1000, dtype=torch.float):
        """
        Args:
            buffer_size: Max number of transitions in buffer.
            state_shape: Shape of the state.
            action_shape: Shape of the action.
            device: Device to place buffer.
            gamma: Discount factor for N-step.
            max_episode_len: Max length of the episode to store.
            dtype: Data type.
        """
        self.buffer_size = buffer_size
        self.max_episodes = buffer_size // max_episode_len
        self.max_episode_len = max_episode_len
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

        self.ep_pointer = 0
        self.cur_episodes = 1
        self.cur_size = 0

        self.actions = torch.empty((self.max_episodes, max_episode_len, *action_shape), dtype=dtype, device=device)
        self.rewards = torch.empty((self.max_episodes, max_episode_len, 1), dtype=dtype, device=device)
        self.dones = torch.empty((self.max_episodes, max_episode_len, 1), dtype=dtype, device=device)
        self.states = torch.empty((self.max_episodes, max_episode_len + 1, *state_shape), dtype=dtype, device=device)
        self.ep_lens = [0] * self.max_episodes

        self.actions_for_std = torch.empty((100, *action_shape), dtype=dtype, device=device)
        self.actions_for_std_cnt = 0

    def append(self, state, action, reward, done, episode_done=None):
        """
        Args:
            state: state.
            action: action.
            reward: reward.
            done: done only if episode ends naturally.
            episode_done: done that can be set to True if time limit is reached.
        """
        self.states[self.ep_pointer, self.ep_lens[self.ep_pointer]].copy_(torch.from_numpy(state))
        self.actions[self.ep_pointer, self.ep_lens[self.ep_pointer]].copy_(torch.from_numpy(action))
        self.rewards[self.ep_pointer, self.ep_lens[self.ep_pointer]] = float(reward)
        self.dones[self.ep_pointer, self.ep_lens[self.ep_pointer]] = float(done)

        self.actions_for_std[self.actions_for_std_cnt % 100].copy_(torch.from_numpy(action))
        self.actions_for_std_cnt += 1

        self.ep_lens[self.ep_pointer] += 1
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
        if episode_done:
            self.ep_pointer = (self.ep_pointer + 1) % self.max_episodes
            self.cur_episodes = min(self.cur_episodes + 1, self.max_episodes)
            self.cur_size -= self.ep_lens[self.ep_pointer]
            self.ep_lens[self.ep_pointer] = 0

    def _inds_to_episodic(self, inds):
        start_inds = np.cumsum([0] + self.ep_lens[:self.cur_episodes - 1])        
        end_inds = start_inds + np.array(self.ep_lens[:self.cur_episodes])
        ep_inds = np.argmin(inds.reshape(-1, 1) >= np.tile(end_inds, (len(inds), 1)), axis=1)
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
            self.states[ep_inds, step_inds + 1]
        )

    def save(self, path: str):
        """
        Args:
            path: Path to pickle file.
        """
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        data = {
            "states": self.states.cpu(), 
            "actions": self.actions.cpu(),
            "rewards": self.rewards.cpu(),
            "dones": self.dones.cpu(),
            "ep_lens": self.ep_lens
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Replay buffer saved to {path}")
        except Exception as e:
            print(f"Failed to save replay buffer: {e}")

    def __len__(self):
        return self.cur_size

    @property
    def num_episodes(self):
        return self.cur_episodes

    def get_last_ep_len(self):
        return self.ep_lens[self.ep_pointer]
