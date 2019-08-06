import gym
#from utils.utils import NormalizedActions

from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker


def create_env_wrapper(config):
    env = config['env'].lower()
    if env == "pendulum-v0":
        return PendulumWrapper(config) #NormalizedActions(gym.make("Pendulum-v0"))
    elif env == "bipedalwalker-v2":
        return BipedalWalker(config)
    else:
        raise ValueError("Unknown environment.")