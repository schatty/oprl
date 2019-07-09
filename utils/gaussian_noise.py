import numpy as np


class GaussianNoiseGenerator:
    def __init__(self, action_dims, action_bound_low, action_bound_high, noise_scale):
        assert np.array_equal(np.abs(action_bound_low), action_bound_high)

        self.action_dims = action_dims
        self.action_bounds = action_bound_high
        self.scale = noise_scale

    def __call__(self):
        noise = np.random.normal(size=self.action_dims) * self.action_bounds * self.scale
        return noise
