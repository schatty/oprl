import gym
import dmc2gym


OPENAI_MUJOCO_PREFIX = [
    "Walker", "HalfCheetah", "Swimmer", "InvertedPendulum", "InvertedDoublePendulum",
    "Hopper", "Humanoid", "Reacher", "Ant"
]


class EnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        open_ai_env = len([pref for pref in OPENAI_MUJOCO_PREFIX if pref in env_name]) > 0
        if open_ai_env:
            self.env = gym.make(self.env_name)
        else:
            domain, task = env_name.split("-")
            self.env = dmc2gym.make(domain_name=domain, task_name=task)

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward

