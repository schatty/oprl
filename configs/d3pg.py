import argparse
import logging
from multiprocessing import Process

import torch.nn as nn

from oprl.algos.ddpg import DDPG, DeterministicPolicy
from oprl.configs.utils import create_logdir
from oprl.distrib.distrib_runner import env_worker, policy_update_worker
from oprl.utils.utils import set_logging

set_logging(logging.INFO)
from oprl.env import make_env as _make_env
from oprl.trainers.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.utils.logger import FileLogger, Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--env", type=str, default="cartpole-balance", help="Name of the environment."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to perform training on."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


# -------- Distrib params -----------

ENV_WORKERS = 4
N_EPISODES = 50  # 500  # Number of episodes each env worker would perform

# -----------------------------------

args = parse_args()


def make_env(seed: int):
    return _make_env(args.env, seed=seed)


env = make_env(seed=0)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
logging.info(f"Env state {STATE_DIM}\tEnv action {ACTION_DIM}")


log_dir = create_logdir(logdir="logs", algo="D3PG", env=args.env, seed=args.seed)
logging.info(f"LOG_DIR: {log_dir}")


def make_logger(seed: int) -> Logger:
    log_dir = create_logdir(logdir="logs", algo="D3PG", env=args.env, seed=seed)
    # TODO: add here actual config
    return FileLogger(log_dir, {})


def make_policy():
    return DeterministicPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_units=(256, 256),
        hidden_activation=nn.ReLU(inplace=True),
        device=args.device,
    )


def make_buffer():
    return EpisodicReplayBuffer(
        buffer_size=int(1_000_000),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        gamma=0.99,
    )


def make_algo():
    logger = make_logger(args.seed)

    algo = DDPG(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        logger=logger,
    )
    return algo


if __name__ == "__main__":
    processes = []

    for i_env in range(ENV_WORKERS):
        processes.append(
            Process(target=env_worker, args=(make_env, make_policy, N_EPISODES, i_env))
        )
    processes.append(
        Process(
            target=policy_update_worker,
            args=(make_algo, make_env, make_buffer, ENV_WORKERS),
        )
    )

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    logging.info("Training OK.")
