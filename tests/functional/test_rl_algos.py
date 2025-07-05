import pytest
import torch

from oprl.algos.ddpg import DDPG
from oprl.algos.sac import SAC
from oprl.algos.td3 import TD3
from oprl.algos.tqc import TQC
from oprl.env import DMControlEnv

rl_algo_classes = [DDPG, SAC, TD3, TQC]


@pytest.mark.parametrize("algo_class", rl_algo_classes)
def test_rl_algo_run(algo_class):
    env = DMControlEnv("walker-walk", seed=0)
    obs, _ = env.reset(env.sample_action())

    algo = algo_class(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    ).create()
    action = algo.exploit(obs)
    assert action.ndim == 1

    action = algo.explore(obs)
    assert action.ndim == 1

    _batch_size = 8
    batch_obs = torch.randn(_batch_size, env.observation_space.shape[0])
    batch_actions = torch.clamp(
        torch.randn(_batch_size, env.action_space.shape[0]), -1, 1
    )
    batch_rewards = torch.randn(_batch_size, 1)
    batch_dones = torch.randint(2, (_batch_size, 1))
    algo.update(batch_obs, batch_actions, batch_rewards, batch_dones, batch_obs)
