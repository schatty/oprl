import os
import argparse
import logging
from multiprocessing import Process

import torch.nn as nn

from oprl.algos.ddpg import DDPG
from oprl.algos.nn_models import DeterministicPolicy
from oprl.distrib.distrib_runner import env_worker, policy_update_worker

from oprl.environment import make_env as _make_env
from oprl.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.logging import (
    LoggerProtocol,
    FileTxtLogger,
    get_logs_path,
)
from oprl.parse_args import parse_args_distrib



# -------- Distrib params -----------

ENV_WORKERS = 4
EPISODES_PER_WORKER = 100  # Number of episodes each env worker would perform

# -----------------------------------

args = parse_args_distrib()

def make_env(seed: int):
    return _make_env(args.env, seed=seed)


env = make_env(seed=0)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
logging.info(f"Env state {STATE_DIM}\tEnv action {ACTION_DIM}")


def make_logger():
    logs_root = os.environ.get("OPRL_LOGS", "logs")
    log_dir = get_logs_path(logdir=logs_root, algo="DistribDDPG", env=args.env, seed=0)
    logger = FileTxtLogger(log_dir)
    logger.copy_source_code()
    return logger


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
        buffer_size_transitions=int(1_000_000),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
    ).create()


def make_algo(logger: LoggerProtocol):
    return DDPG(
        logger=logger,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
    ).create()


if __name__ == "__main__":
    processes = []

    for i_env in range(ENV_WORKERS):
        processes.append(
            Process(target=env_worker, args=(make_env, make_policy, EPISODES_PER_WORKER, i_env))
        )
    processes.append(
        Process(
            target=policy_update_worker,
            args=(make_algo, make_env, make_buffer, make_logger, ENV_WORKERS),
        )
    )

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    logging.info("Training OK.")
