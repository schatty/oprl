import pickle
import time
from multiprocessing import Process
import pika

import torch


def env_worker(make_env, make_policy):
    env = make_env("walker-walk", seed=0)
    print("Env created.")

    policy = make_policy()
    print("Policy created.")

    connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="localhost")
    )
    channel = connection.channel()

    channel.queue_declare(queue="hello")
    print("Queue created.")

    episodes = []
    
    for _ in range(1):
        ep_step = 0
        start_steps = 1000
        episode = []

        state, _ = env.reset()
        for env_step in range(1000):
            ep_step += 1
            if env_step <= start_steps:
                action = env.sample_action()
            else:
                state = torch.tensor(state).unsqueeze(0)
                # TODO: move state to device
                action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append([state, action, reward, terminated, next_state])

            if terminated or truncated:
                break
            state = next_state

        episode_serialized = pickle.dumps(episode)
        channel.basic_publish(exchange="", routing_key="hello", body=episode_serialized)
        print("Episode sent!")

    print("Episode by env worker is done.")


def policy_update_worker(make_algo, make_buffer):
    algo = make_algo()
    print("Algo created.")
    buffer = make_buffer()
    print("Buffer created.")

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()
    print("Learner queue created.")

    batch_size = 128

    channel.queue_declare(queue="hello")

    def callback(ch, method, properties, body):
        print(f"[x] Received chunk")
        data = pickle.loads(body)
        print("Looking at data: ", type(data))
        print(len(data))
        print("Callback end.")

    for _ in range(1):
        if len(buffer) < batch_size:
            continue
        batch = buffer.sample(batch_size)
        algo.update(batch)
    print("Update by policy update worker done.")

    # channel.basic_consume(queue="hello", on_message_callback=callback, auto_ack=True)
    # print("Consuming queue is done")

    print("basic_get")
    method_frame, header_frame, body = channel.basic_get(queue="hello")

    if method_frame:
        print(method_frame)
        data = pickle.loads(body)
        print("Looking at data: ", type(data))
        print(len(data))
        print("Got body: ", type(body))
        data = pickle.loads(body)
        print("Looking at data: ", type(data))
        print(len(data))

