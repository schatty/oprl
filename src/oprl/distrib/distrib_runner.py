import pickle
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch as t
import torch.nn as nn
import pika

from oprl.logging import create_stdout_logger


logger = create_stdout_logger()


class Queue:
    def __init__(self, name: str, host: str = "localhost"):
        self._name = name

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = connection.channel()
        self.channel.queue_declare(queue=name)

    def push(self, data) -> None:
        self.channel.basic_publish(exchange="", routing_key=self._name, body=data)

    def pop(self) -> bytes | None:
        method_frame, header_frame, body = self.channel.basic_get(queue=self._name)
        if method_frame:
            self.channel.basic_ack(method_frame.delivery_tag)
            return body
        return None


def env_worker(make_env, make_policy, n_episodes, id_worker):
    env = make_env(seed=0)
    logger.info("Env created.")

    policy = make_policy()
    logger.info("Policy created.")

    q_env = Queue(f"env_{id_worker}")
    q_policy = Queue(f"policy_{id_worker}")
    logger.info("Queue created.")

    total_env_step = 0
    # TODO: Move parameter to config
    start_steps = 1000
    for i_ep in range(n_episodes):
        print("Running episode: ", i_ep)
        if i_ep % 10 == 0:
            logger.info(f"AGENT {id_worker} EPISODE {i_ep}")

        episode = []
        state, _ = env.reset()
        # TODO: Move parameter to config
        for env_step in range(1000):
            if total_env_step <= start_steps:
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


def save_policy(policy: nn.Module, save_path: Path):
    save_path.parents[0].mkdir(exist_ok=True)
    t.save(
        policy,
        save_path
    )


def policy_update_worker(make_algo, make_env_test, make_buffer, make_logger, n_workers):
    scalar_logger = make_logger()
    algo = make_algo(scalar_logger)
    logger.info("Algo created.")
    buffer = make_buffer()
    logger.info("Buffer created.")

    q_envs = []
    q_policies = []
    for i_env in range(n_workers):
        q_envs.append(Queue(f"env_{i_env}"))
        q_policies.append(Queue(f"policy_{i_env}"))
    logger.info("Learner queue created.")

    batch_size = 128

    logger.info("Warming up the learner...")
    time.sleep(2.0)

    for i_epoch in count(0):
        logger.info(f"Epoch: {i_epoch}")
        n_waits = 0
        for i_env in range(n_workers):
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
                    if n_waits == 10:
                        logger.info("Learner tired to wait, exiting...")
                        return
                    continue

        # TODO: Remove hardcoded value
        if i_epoch > 16:
            for i in range(1000 * 4):
                batch = buffer.sample(batch_size)
                algo.update(*batch)
                if i % int(1000) == 0:
                    logger.info(f"\tUpdating {i}")

        policy_state_dict = algo.get_policy_state_dict()

        policy_serialized = pickle.dumps(policy_state_dict)
        for i_env in range(n_workers):
            q_policies[i_env].push(policy_serialized)

 
        if i_epoch > 0 and i_epoch % 10 == 0:
            mean_reward = evaluate(algo, make_env_test)
            logger.info(f"Evaluating policy [epoch {i_epoch}]: {mean_reward}")
            algo.logger.log_scalar("trainer/ep_reward", mean_reward, i_epoch)

            save_policy(
                policy=algo.actor,
                save_path=algo.logger.log_dir / "weights" / f"epoch_{i_epoch}.w"
            )
            logger.info(f"Weights saved.")

    logger.info("Update by policy update worker done.")


def evaluate(algo, make_env_test, num_eval_episodes: int = 5, seed: int = 0):
    returns = []
    for i_ep in range(num_eval_episodes):
        env_test = make_env_test(seed * 100 + i_ep)
        state, _ = env_test.reset()

        episode_return = 0.0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = algo.exploit(state)
            state, reward, terminated, truncated, _ = env_test.step(action)
            episode_return += reward

        returns.append(episode_return)

    mean_return = np.mean(returns)
    return mean_return
