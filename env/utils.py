import gym
#from utils.utils import NormalizedActions

from .pendulum import PendulumWrapper


def create_env_wrapper(config):
    env = config['env'].lower()
    if env == "pendulum-v0":
        return PendulumWrapper(config) #NormalizedActions(gym.make("Pendulum-v0"))
    else:
        raise ValueError("Unknown environment.")