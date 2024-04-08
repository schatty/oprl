import logging
import pickle
import time
from itertools import count
from multiprocessing import Process

import numpy as np
import pika
import torch


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
    logging.info("Env created.")

    policy = make_policy()
    logging.info("Policy created.")

    q_env = Queue(f"env_{id_worker}")
    q_policy = Queue(f"policy_{id_worker}")
    logging.info("Queue created.")

    episodes = []

    total_env_step = 0
    # TODO: Move parameter to config
    start_steps = 1000
    for i_ep in range(n_episodes):
        if i_ep % 10 == 0:
            logging.info(f"AGENT {id_worker} EPISODE {i_ep}")

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
                logging.info("Waiting for the policy..")
                time.sleep(2.0)
                continue

            policy.load_state_dict(pickle.loads(data))
            break

    logging.info("Episode by env worker is done.")


def policy_update_worker(make_algo, make_env_test, make_buffer, n_workers):
    algo = make_algo()
    logging.info("Algo created.")
    buffer = make_buffer()
    logging.info("Buffer created.")

    q_envs = []
    q_policies = []
    for i_env in range(n_workers):
        q_envs.append(Queue(f"env_{i_env}"))
        q_policies.append(Queue(f"policy_{i_env}"))
    logging.info("Learner queue created.")

    batch_size = 128

    logging.info("Warming up the learner...")
    time.sleep(2.0)

    for i_epoch in count(0):
        logging.info(f"Epoch: {i_epoch}")
        n_waits = 0
        for i_env in range(n_workers):
            while True:
                data = q_envs[i_env].pop()
                if data:
                    episode = pickle.loads(data)
                    buffer.add_episode(episode)
                    break
                else:
                    logging.info("Waiting for the env data...")
                    # TODO: not optimal wait for each queue
                    time.sleep(1)
                    n_waits += 1
                    if n_waits == 10:
                        logging.info("Learner tired to wait, exiting...")
                        return
                    continue

        # TODO: Remove hardcoded value
        if i_epoch > 16:
            for i in range(1000 * 4):
                batch = buffer.sample(batch_size)
                algo.update(*batch)
                if i % int(1000) == 0:
                    logging.info(f"\tUpdating {i}")

        policy_state_dict = algo.get_policy_state_dict()

        policy_serialized = pickle.dumps(policy_state_dict)
        for i_env in range(n_workers):
            q_policies[i_env].push(policy_serialized)

        if True:
            mean_reward = evaluate(algo, make_env_test)
            algo.logger.log_scalar("trainer/ep_reward", mean_reward, i_epoch)

    logging.info("Update by policy update worker done.")


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
