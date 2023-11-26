import os
from datetime import datetime
import time
from multiprocessing import Process
import argparse

import torch
import torch.nn as nn

from env import make_env, DMControlEnv
from algos.ddpg import DDPG, DeterministicPolicy
from distrib.distrib_runner import env_worker, policy_update_worker
from trainers.buffers.episodic_buffer import EpisodicReplayBuffer
from utils.logger import Logger
print("Imports ok.")


def parse_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--env", type=str, default="cartpole-balance", help="Name of the environment.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to perform training on.")
    return parser.parse_args()


args = parse_args()

def make_env(seed: int):
    """
    Args:
        name: Environment name.
    """
    return DMControlEnv(args.env, seed=seed)

env = make_env(seed=0)

STATE_SHAPE = env.observation_space.shape
ACTION_SHAPE = env.action_space.shape
print("STATE ACTION SHAPE: ", STATE_SHAPE, ACTION_SHAPE)


def make_policy():
    return DeterministicPolicy(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        hidden_units=[256, 256],
        hidden_activation=nn.ReLU(inplace=True)
    )

def make_buffer():
    buffer = EpisodicReplayBuffer(
        buffer_size=int(100_000),
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device="cpu",
        gamma=0.99
    )
    return buffer


def make_algo():
    time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    log_dir = os.path.join(
        "logs_debug", "DDPG", 
        f"DDPG-env_ENV-seedSEED-{time}")
    print("LOGDIR: ", log_dir)
    logger = Logger(log_dir, {})

    algo = DDPG(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device="cpu",
        seed=0,
        logger=logger
    )
    return algo


if __name__ == "__main__":
    ENV_WORKERS = 2

    seed = 0

    processes = []

    for i_env in range(ENV_WORKERS):
        processes.append(Process(target=env_worker, args=(make_env, make_policy, i_env)))
    processes.append(Process(target=policy_update_worker, args=(make_algo, make_env, make_buffer, ENV_WORKERS)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("OK.")
