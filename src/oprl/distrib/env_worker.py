import pickle
import time
from typing import Callable

from oprl.algos.protocols import PolicyProtocol
from oprl.environment.protocols import EnvProtocol
from oprl.logging import create_stdout_logger
from oprl.runners.config import DistribConfig
from oprl.distrib.queue import Queue


logger = create_stdout_logger()


def run_env_worker(
    make_env: Callable[[int], EnvProtocol],
    make_policy: Callable[[], PolicyProtocol],
    config: DistribConfig,
    id_worker: int,
) -> None:
    env = make_env(seed=0)
    logger.info("Env created.")

    policy = make_policy()
    logger.info("Policy created.")

    q_env = Queue(f"env_{id_worker}")
    q_policy = Queue(f"policy_{id_worker}")
    logger.info("Queue created.")

    total_env_step = 0
    for i_ep in range(config.episodes_per_worker):
        print("Running episode: ", i_ep)
        if i_ep % 10 == 0:
            logger.info(f"AGENT {id_worker} EPISODE {i_ep}")

        episode = []
        state, _ = env.reset()
        for _ in range(config.episode_length):
            if total_env_step <= config.warmup_env_steps:
                action = env.sample_action()
            else:
                action = policy.explore(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append([state, action, reward, terminated, next_state])

            if terminated or truncated:
                break
            state = next_state
            total_env_step += 1

        q_env.push(pickle.dumps(episode))

        while True:
            data = q_policy.pop()
            if data is None:
                logger.info("Waiting for the policy..")
                time.sleep(2.0)
                continue
            policy.load_state_dict(pickle.loads(data))
            break

    logger.info("Episode by env worker is done.")

