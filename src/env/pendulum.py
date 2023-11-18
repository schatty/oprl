import numpy as np
from .env_wrapper import EnvWrapper


class PendulumWrapper(EnvWrapper):
    def __init__(self, config):
        EnvWrapper.__init__(self, config['env'])
        self.config = config

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward/100.0