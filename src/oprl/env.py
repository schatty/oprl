from collections import OrderedDict

import numpy as np
from dm_control import suite



class DMControlEnv:

    def __init__(self, env: str, seed: int):
        domain, task = env.split("-")
        self.random_state = np.random.RandomState(seed)
        self.env = suite.load(domain, task, task_kwargs={"random": self.random_state})

        self._render_width = 200
        self._render_height = 200
        self._camera_id = 0

    def reset(self, *args, **kwargs):
        obs = self._flat_obs(self.env.reset().observation)
        return obs, {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flat_obs(time_step.observation)

        terminated = False
        truncated = self.env._step_count >= self.env._step_limit

        return obs, time_step.reward, terminated, truncated, {}

    def sample_action(self):
        spec = self.env.action_spec()
        action = self.random_state.uniform(spec.minimum, spec.maximum, spec.shape) 
        return action

    @property
    def observation_space(self):
        return np.zeros(sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values()))

    @property
    def action_space(self):
        return np.zeros(self.env.action_spec().shape[0])

    def render(self):
        """
        returned shape: [1, W, H, C]
        """
        img = self.env.physics.render(camera_id=self._camera_id,
                                       height=self._render_width, width=self._render_width)
        img = img.astype(np.uint8)
        return np.expand_dims(img, 0)

    def _flat_obs(self, obs: OrderedDict):
        obs_flatten = []
        for _, o in obs.items():
            if len(o.shape) == 0:
                obs_flatten.append(np.array([o]))
            elif len(o.shape) == 2 and o.shape[1] > 1:
                obs_flatten.append(o.flatten())
            else:
                obs_flatten.append(o)
        return np.concatenate(obs_flatten)


def make_env(name: str, seed: int):
    """
    Args:
        name: Environment name.
    """
    return DMControlEnv(name, seed=seed)

