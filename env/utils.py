import gym
from utils.utils import NormalizedActions


def create_env_wrapper(env):
    env = env.lower()
    if env == "pendulum-v0":
        return NormalizedActions(gym.make("Pendulum-v0"))
    else:
        raise ValueError("Unknown environment.")