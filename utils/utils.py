import numpy as np
import gym
import imageio
from glob import glob
import os
import yaml


class OUNoise(object):
    def __init__(self, dim, low, high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10_000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = dim
        self.low = low
        self.high = high

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        action = action.cpu().detach().numpy()
        return np.clip(action + ou_state, self.low, self.high)


def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
    Returns: None
    """
    batch_sort = lambda s: int(s[s.rfind('/')+1:s.rfind('.')])
    image_paths = sorted(glob(os.path.join(source_dir, "*.png")),
                         key=batch_sort)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(output, images)


def read_config(path):
    """
    Return python dict from .yml file.

    Args:
        path (str): path to the .yml config.

    Returns (dict): configuration object.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Load environment from gym to set its params
    env = gym.make(cfg['env'])
    cfg['state_dims'] = env.observation_space.shape[0]
    cfg['state_bound_low'] = env.observation_space.low
    cfg['state_bound_high'] = env.observation_space.high
    cfg['action_dims'] = env.action_space.shape[0]
    cfg['action_bound_low'] = env.action_space.low
    cfg['action_bound_high'] = env.action_space.high
    del env

    return cfg