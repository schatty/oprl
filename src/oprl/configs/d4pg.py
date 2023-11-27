import logging
import os
from datetime import datetime
import time
from multiprocessing import Process
import argparse

import torch
import torch.nn as nn

from oprl.env import make_env, DMControlEnv
from oprl.algos.ddpg import DDPG, DeterministicPolicy
from oprl.distrib.distrib_runner import env_worker, policy_update_worker
from oprl.trainers.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.utils.logger import Logger
from oprl.configs.utils import create_logdir

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--env", type=str, default="cartpole-balance", help="Name of the environment.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to perform training on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


# -------- Distrib params -----------

ENV_WORKERS = 4
N_EPISODES = 500  # Number of episodes each env worker would perform

# -----------------------------------

args = parse_args()

def make_env(seed: int):
    return DMControlEnv(args.env, seed=seed)

env = make_env(seed=0)
STATE_SHAPE = env.observation_space.shape
ACTION_SHAPE = env.action_space.shape
logging.info(f"Env state {STATE_SHAPE}\tEnv action {ACTION_SHAPE}")


log_dir = create_logdir(logdir="logs", algo="D3PG", env=args.env, seed=args.seed)
print("LOG_DIR: ", log_dir)


def make_policy():
    return DeterministicPolicy(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        hidden_units=[256, 256],
        hidden_activation=nn.ReLU(inplace=True),
        device=args.device,
    )


def make_buffer():
    return EpisodicReplayBuffer(
        buffer_size=int(1_000_000),
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device=args.device,
        gamma=0.99,
    )


def make_algo():
    logger = Logger(log_dir, {})

    algo = DDPG(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device=args.device,
        seed=args.seed,
        logger=logger
    )
    return algo


if __name__ == "__main__":

    processes = []

    for i_env in range(ENV_WORKERS):
        processes.append(Process(target=env_worker, args=(make_env, make_policy, N_EPISODES, i_env)))
    processes.append(Process(target=policy_update_worker, args=(make_algo, make_env, make_buffer, ENV_WORKERS)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("OK.")
