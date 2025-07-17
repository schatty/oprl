import pytest
import torch
import numpy.typing as npt

from oprl.algos.protocols import AlgorithmProtocol
from oprl.algos.ddpg import DDPG
from oprl.algos.sac import SAC
from oprl.algos.td3 import TD3
from oprl.algos.tqc import TQC
from oprl.environment import DMControlEnv
from oprl.environment.protocols import EnvProtocol
from oprl.logging import FileTxtLogger


rl_algo_classes: list[type[AlgorithmProtocol]] = [DDPG, SAC, TD3, TQC]

def _run_common_test(algo: AlgorithmProtocol, env: EnvProtocol, obs: npt.NDArray) -> None:
    action = algo.actor.exploit(obs)
    assert action.ndim == 1

    action = algo.actor.explore(obs)
    assert action.ndim == 1

    _batch_size = 8
    batch_obs = torch.randn(_batch_size, env.observation_space.shape[0])
    batch_actions = torch.clamp(
        torch.randn(_batch_size, env.action_space.shape[0]), -1, 1
    )
    batch_rewards = torch.randn(_batch_size, 1)
    batch_dones = torch.randint(2, (_batch_size, 1))
    algo.update(batch_obs, batch_actions, batch_rewards, batch_dones, batch_obs)


@pytest.mark.parametrize("algo_class", rl_algo_classes)
def test_ddpg_td3_tqc(algo_class: type[AlgorithmProtocol]) -> None:
    env = DMControlEnv("walker-walk", seed=0)
    # TODO: Change to mocked logger
    logger = FileTxtLogger(".")
    obs, _ = env.reset()
    algo = algo_class(
        logger=logger,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    ).create()
    _run_common_test(algo, env, obs)


@pytest.mark.parametrize("tune_alpha", [True, False])
def test_sac(tune_alpha: bool) -> None:
    env = DMControlEnv("walker-walk", seed=0)
    # TODO: Change to mocked logger
    logger = FileTxtLogger(".")
    obs, _ = env.reset()
    algo = SAC(
        logger=logger,
        tune_alpha=tune_alpha,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    ).create()
    _run_common_test(algo, env, obs)
