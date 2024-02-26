import pytest

from oprl.env import SafetyGym

dm_control_envs = [
    "SafetyPointGoal1-v0",
    "SafetyPointButton1-v0",
    "SafetyPointPush1-v0",
    "SafetyPointCircle1-v0",
    "SafetyCarGoal1-v0",
    "SafetyCarButton1-v0",
    "SafetyCarPush1-v0",
    "SafetyCarCircle1-v0",
    "SafetyAntGoal1-v0",
    "SafetyAntButton1-v0",
    "SafetyAntPush1-v0",
    "SafetyAntCircle1-v0",
    "SafetyDoggoGoal1-v0",
    "SafetyDoggoButton1-v0",
    "SafetyDoggoPush1-v0",
    "SafetyDoggoCircle1-v0",
]


@pytest.mark.parametrize("env_name", dm_control_envs)
def test_dm_control_envs(env_name: str):
    env = SafetyGym(env_name, seed=0)
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    assert isinstance(info, dict), "Info is expected to be a dict"

    rand_action = env.sample_action()
    assert rand_action.ndim == 1

    next_obs, reward, terminated, truncated, info = env.step(rand_action)
    assert next_obs.ndim == 1, "Expected 1-dimensional array as observation"
    assert isinstance(reward, float), "Reward is epxected to be a single float value"
    assert isinstance(
        terminated, bool
    ), "Terminated is expected to be a single bool value"
    assert isinstance(
        truncated, bool
    ), "Truncated is expected to be a single bool value"
    assert isinstance(info, dict), "Info is expected to be a dict"
    assert "cost" in info, "Info is expected to contain cost key"
