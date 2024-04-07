from collections import OrderedDict

import numpy as np
from dm_control import suite


class SafetyGym:
    def __init__(self, env_name: str, seed: int):
        import safety_gymnasium as gym

        self._env = gym.make(env_name)
        self._seed = seed

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        info["cost"] = cost
        return obs, reward, terminated, truncated, info

    def reset(self):
        obs, info = self._env.reset(seed=self._seed)
        self._env.step(self._env.action_space.sample())
        return obs, info

    def sample_action(self):
        return self._env.action_space.sample()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def env_family(self) -> str:
        return "safety_gymnasium"


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
        return np.zeros(
            sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values())
        )

    @property
    def action_space(self):
        return np.zeros(self.env.action_spec().shape[0])

    def render(self):
        """
        returned shape: [1, W, H, C]
        """
        img = self.env.physics.render(
            camera_id=self._camera_id,
            height=self._render_width,
            width=self._render_width,
        )
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
        return np.concatenate(obs_flatten, dtype="float32")

    @property
    def env_family(self) -> str:
        return "dm_control"


ENV_MAPPER = {
    "dm_control": set(
        [
            "acrobot-swingup",
            "ball_in_cup-catch",
            "cartpole-balance",
            "cartpole-swingup",
            "cheetah-run",
            "finger-spin",
            "finger-turn_easy",
            "finger-turn_hard",
            "fish-upright",
            "fish-swim",
            "hopper-stand",
            "hopper-hop",
            "humanoid-stand",
            "humanoid-walk",
            "humanoid-run",
            "pendulum-swingup",
            "point_mass-easy",
            "reacher-easy",
            "reacher-hard",
            "swimmer-swimmer6",
            "swimmer-swimmer15",
            "walker-stand",
            "walker-walk",
            "walker-run",
        ]
    ),
    "safety_gymnasium": set(
        [
            "SafetyPointGoal1-v0",
            "SafetyPointGoal2-v0",
            "SafetyPointButton1-v0",
            "SafetyPointButton2-v0",
            "SafetyPointPush1-v0",
            "SafetyPointPush2-v0",
            "SafetyPointCircle1-v0",
            "SafetyPointCircle2-v0",
            "SafetyCarGoal1-v0",
            "SafetyCarGoal2-v0",
            "SafetyCarButton1-v0",
            "SafetyCarButton2-v0",
            "SafetyCarPush1-v0",
            "SafetyCarPush2-v0",
            "SafetyCarCircle1-v0",
            "SafetyCarCircle2-v0",
            "SafetyAntGoal1-v0",
            "SafetyAntGoal2-v0",
            "SafetyAntButton1-v0",
            "SafetyAntButton2-v0",
            "SafetyAntPush1-v0",
            "SafetyAntPush2-v0",
            "SafetyAntCircle1-v0",
            "SafetyAntCircle2-v0",
            "SafetyDoggoGoal1-v0",
            "SafetyDoggoGoal2-v0",
            "SafetyDoggoButton1-v0",
            "SafetyDoggoButton2-v0",
            "SafetyDoggoPush1-v0",
            "SafetyDoggoPush2-v0",
            "SafetyDoggoCircle1-v0",
            "SafetyDoggoCircle2-v0",
        ]
    ),
}


def make_env(name: str, seed: int):
    """
    Args:
        name: Environment name.
    """
    for env_type, env_set in ENV_MAPPER.items():
        if name in env_set:
            if env_type == "dm_control":
                return DMControlEnv(name, seed=seed)
            elif env_type == "safety_gymnasium":
                return SafetyGym(name, seed=seed)
    else:
        raise ValueError(f"Unsupported environment: {name}")
