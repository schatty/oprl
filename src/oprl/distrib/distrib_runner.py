import pickle
import time
from multiprocessing import Process
import pika
from itertools import count

import torch
import numpy as np


class Queue:
    def __init__(self, name: str, host: str = "localhost"):
        self._name = name

        connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host)
        )
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
    print("Env created.")

    policy = make_policy()
    print("Policy created.")

    q_env = Queue(f"env_{id_worker}")
    q_policy = Queue(f"policy_{id_worker}")
    print("Queue created.")

    episodes = []
    
    total_env_step = 0
    start_steps = 1000
    for i_ep in range(n_episodes):
        if i_ep % 10 == 0:
            print(f"AGENT {id_worker} EPISODE {i_ep}")

        episode = []
        state, _ = env.reset()
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
        # print(f"Episode {i_ep} sent!")
        
        while True:
            data = q_policy.pop()
            if data is None:
                print("Waiting for the policy..")
                time.sleep(2.0)
                continue
        
            policy.load_state_dict(pickle.loads(data))
            # print("New policy in agent loaded")
            break

    print("Episode by env worker is done.")


def policy_update_worker(make_algo, make_env_test, make_buffer, n_workers):
    algo = make_algo()
    print("Algo created.")
    buffer = make_buffer()
    print("Buffer created.")

    q_envs = []
    q_policies = []
    for i_env in range(n_workers):
        q_envs.append(Queue(f"env_{i_env}"))
        q_policies.append(Queue(f"policy_{i_env}"))
    print("Learner queue created.")

    batch_size = 128

    print("Warming up the learner...")
    time.sleep(2.0)

    for i_epoch in count(0):
        print("Epoch ", i_epoch)
        n_waits = 0
        for i_env in range(n_workers):
            while True:
                data = q_envs[i_env].pop()
                if data:
                    episode = pickle.loads(data)
                    buffer.add_episode(episode)
                    # print("Episode added to buffer OK.")
                    break
                else:
                    print("Waiting for the env data...")
                    # TODO: not optimal wait for each queue
                    time.sleep(1)
                    n_waits += 1
                    if n_waits == 10:
                        print("Learner tired to wait, exiting...")
                        return
                    continue

        if i_epoch > 16:
            for i in range(1000 * 4):
                batch = buffer.sample(batch_size)
                algo.update(batch)
                if i % int(1000) == 0:
                    print(f"\tUpdating {i}")
            # print("Upadate performed OK.")
        
        policy_state_dict = algo.get_policy_state_dict()

        policy_serialized = pickle.dumps(policy_state_dict)
        for i_env in range(n_workers):
            q_policies[i_env].push(policy_serialized)
        # print("Pushing policy")
    
        if True:
            mean_reward = evaluate(algo, make_env_test)
            algo.logger.log_scalar("trainer/ep_reward", mean_reward, i_epoch)

    print("Update by policy update worker done.")


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

