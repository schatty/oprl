import os
import numpy as np
import pickle


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), save_dir="~"):
		self.max_size = max_size
		self.save_dir = save_dir
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

	def add(self, state, action, reward, next_state, done, *args):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			self.state[ind],
			self.action[ind],
			self.reward[ind],
			self.next_state[ind],
			self.not_done[ind]
		)

	def __len__(self):
		return self.size

	def dump(self):
		fn = os.path.join(self.save_dir, 'replay_buffer.pkl')
		buffer = {
			'state': self.state,
			'action': self.action,
			'reward': self.reward,
			'next_state': self.next_state,
			'not_done': self.not_done
		}
		with open(fn, 'wb') as f:
			pickle.dump(buffer, f)
		print("Buffer dumped.")