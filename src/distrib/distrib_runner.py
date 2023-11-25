import pickle
import time
from multiprocessing import Process
import pika

import torch


class Queue:
    def __init__(self, name: str, host: str = "localhost"):
        self._name = name

        connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host)
        )
        self.channel = connection.channel()
        self.channel.queue_declare(queue=name)

    def push(self, data) -> None:
        print("Pushing type of data: ", type(data))
        self.channel.basic_publish(exchange="", routing_key=self._name, body=data)

    def pop(self) -> bytes | None:
        method_frame, header_frame, body = self.channel.basic_get(queue=self._name)
        if method_frame:
            self.channel.basic_ack(method_frame.delivery_tag)
            return body
        return None


def env_worker(make_env, make_policy):
    env = make_env("walker-walk", seed=0)
    print("Env created.")

    policy = make_policy()
    print("Policy created.")

    q = Queue("env_0")
    print("Queue created.")

    episodes = []
    
    for i_ep in range(3):
        ep_step = 0
        start_steps = 1000
        episode = []

        state, _ = env.reset()
        for env_step in range(1000):
            ep_step += 1
            if env_step <= start_steps:
                action = env.sample_action()
            else:
                action = policy.exploit(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append([state, action, reward, terminated, next_state])

            if terminated or truncated:
                break
            state = next_state

        q.push(pickle.dumps(episode))
        print(f"Episode {i_ep} sent!")

    print("Episode by env worker is done.")


def policy_update_worker(make_algo, make_buffer):
    algo = make_algo()
    print("Algo created.")
    buffer = make_buffer()
    print("Buffer created.")

    q = Queue("env_0")
    print("Learner queue created.")

    batch_size = 128

    print("Warming up the learner...")
    time.sleep(2.0)

    while True:
        print("basic_get")
        data = q.pop()
        if data:
            episode = pickle.loads(data)
            buffer.add_episode(episode)
            print("Episode added to buffer OK.")

        time.sleep(1)

        if len(buffer) < batch_size:
            print("Buffer too small, not performing update")
            continue
        batch = buffer.sample(batch_size)
        algo.update(batch)
        print("Upadate performed OK.")
        
        # TODO: Send updated policy to agents
        time.sleep(1)
    print("Update by policy update worker done.")

