import pickle
import time
from itertools import count
from pathlib import Path
from typing import Callable

import numpy as np
import torch as t
import torch.nn as nn

from oprl.algos.protocols import AlgorithmProtocol
from oprl.environment.protocols import EnvProtocol
from oprl.logging import create_stdout_logger, LoggerProtocol
from oprl.runners.config import DistribConfig
from oprl.buffers.protocols import ReplayBufferProtocol
from oprl.distrib.queue import Queue


logger = create_stdout_logger()


def run_policy_update_worker(
    make_algo: Callable[[LoggerProtocol], AlgorithmProtocol],
    make_env_test: Callable[[int], EnvProtocol],
    make_buffer: Callable[[], ReplayBufferProtocol],
    make_logger: Callable[[], LoggerProtocol],
    config: DistribConfig,
) -> None:
    scalar_logger = make_logger()
    algo = make_algo(scalar_logger)
    logger.info("Algo created.")
    buffer = make_buffer()
    logger.info("Buffer created.")

    q_envs = []
    q_policies = []
    for i_env in range(config.num_env_workers):
        q_envs.append(Queue(f"env_{i_env}"))
        q_policies.append(Queue(f"policy_{i_env}"))
    logger.info("Learner queue created.")

    logger.info("Warming up the learner...")
    time.sleep(2.0)

    for i_epoch in count(0):
        logger.info(f"Epoch: {i_epoch}")
        n_waits = 0
        for i_env in range(config.num_env_workers):
            while True:
                data = q_envs[i_env].pop()
                if data:
                    episode = pickle.loads(data)
                    buffer.add_episode(episode)
                    break
                else:
                    logger.info("Waiting for the env data...")
                    # TODO: not optimal wait for each queue
                    time.sleep(1)
                    n_waits += 1
                    if n_waits == config.learner_num_waits:
                        logger.info("Learner is not receiving data, exiting...")
                        return
                    continue

        if i_epoch > config.warmup_epochs:
            for i in range(config.episode_length * config.num_env_workers):
                batch = buffer.sample(config.batch_size)
                algo.update(*batch)
                if i % int(1000) == 0:
                    logger.info(f"\tUpdating {i}")

        policy_state_dict = algo.get_policy_state_dict()

        policy_serialized = pickle.dumps(policy_state_dict)
        for i_env in range(config.num_env_workers):
            q_policies[i_env].push(policy_serialized)

 
        if i_epoch > 0 and i_epoch % 10 == 0:
            mean_reward = evaluate(algo, make_env_test)
            logger.info(f"Evaluating policy [epoch {i_epoch}]: {mean_reward}")
            algo.logger.log_scalar("trainer/ep_reward", mean_reward, i_epoch)
            save_policy(
                policy=algo.actor,
                save_path=algo.logger.log_dir / "weights" / f"epoch_{i_epoch}.w"
            )
            logger.info("Weights saved.")

    logger.info("Update by policy update worker done.")


def save_policy(policy: nn.Module, save_path: Path) -> None:
    save_path.parents[0].mkdir(exist_ok=True)
    t.save(
        policy,
        save_path
    )


def evaluate(
    algo: AlgorithmProtocol,
    make_env_test: Callable[[int], EnvProtocol],
    num_eval_episodes: int = 5,
    seed: int = 0
) -> float:
    returns = []
    for i_ep in range(num_eval_episodes):
        env_test = make_env_test(seed * 100 + i_ep)
        state, _ = env_test.reset()

        episode_return = 0.0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = algo.actor.exploit(state)
            state, reward, terminated, truncated, _ = env_test.step(action)
            episode_return += reward
        returns.append(episode_return)

    return np.mean(returns)
