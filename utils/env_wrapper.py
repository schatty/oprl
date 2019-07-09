'''
## Env Wrapper ##
# A wrapper around the OpenAI Gym environments. Each environment requires its own custom wrapper as the preprocessing required differs by environment.
@author: Mark Sinton (msinto93@gmail.com)
'''

import gym
from params import train_params


class EnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()


class PendulumWrapper(EnvWrapper):
    def __init__(self, env_name):
        EnvWrapper.__init__(self, env_name)

        # State
        # Type: Box(3)
        # Num    Observation    Min    Max
        #  0     cos(theta)    -1.0    1.0
        #  1     sin(theta)    -1.0    1.0
        #  2     theta dot     -8.0    8.0

        # Action
        # Type: Box(1)
        # Num    Action           Min    Max
        #  0     Joint effort    -2.0    2.0

        # Reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
        # Theta lies between -pi and pi
        # Goal is to remain at zero angle (theta=0), with least rotational velocity (theta_dot=0) and least effort (action=0).
        # Max reward is therefore 0.0.
        # Min reward occurs at max angle (theta_max), max rotational velocity (theta_dot_max) and max effort (action_max) - approx -16.27

    def normalise_state(self, state):
        # Normalise state values to [-1, 1] range
        return state / train_params.STATE_BOUND_HIGH

    def normalise_reward(self, reward):
        # Normalise reward values
        return reward / 100.0
