import numpy as np
from .env_wrapper import EnvWrapper


class PendulumWrapper(EnvWrapper):
    def __init__(self, config):
        EnvWrapper.__init__(self, config['env'])
        self.config = config

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
        return state

    def normalise_reward(self, reward):
        return reward/100.0