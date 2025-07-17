import os
import logging

import torch.nn as nn

from oprl.algos.ddpg import DDPG
from oprl.algos.nn_models import DeterministicPolicy
from oprl.algos.protocols import AlgorithmProtocol, PolicyProtocol
from oprl.buffers.protocols import ReplayBufferProtocol
from oprl.environment import make_env as _make_env
from oprl.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.logging import (
    LoggerProtocol,
    FileTxtLogger,
    get_logs_path,
)
from oprl.parse_args import parse_args_distrib
from oprl.runners.config import DistribConfig
from oprl.runners.train_distrib import run_distrib_training
from oprl.distrib.env_worker import run_env_worker
from oprl.distrib.policy_update_worker import run_policy_update_worker


config = DistribConfig(
    batch_size=128,
    num_env_workers=4,
    episodes_per_worker=100,
    warmup_epochs=16,
    episode_length=1000,
    learner_num_waits=10,
)


args = parse_args_distrib()

def make_env(seed: int):
    return _make_env(args.env, seed=seed)


env = make_env(seed=0)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
logging.info(f"Env state {STATE_DIM}\tEnv action {ACTION_DIM}")


def make_logger() -> LoggerProtocol:
    logs_root = os.environ.get("OPRL_LOGS", "logs")
    log_dir = get_logs_path(logdir=logs_root, algo="DistribDDPG", env=args.env, seed=0)
    logger = FileTxtLogger(log_dir)
    logger.copy_source_code()
    return logger


def make_policy() -> PolicyProtocol:
    return DeterministicPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_units=(256, 256),
        hidden_activation=nn.ReLU(inplace=True),
        device=args.device,
    )


def make_replay_buffer() -> ReplayBufferProtocol:
    return EpisodicReplayBuffer(
        buffer_size_transitions=int(1_000_000),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
    ).create()


def make_algo(logger: LoggerProtocol) -> AlgorithmProtocol:
    return DDPG(
        logger=logger,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
    ).create()


if __name__ == "__main__":
    run_distrib_training(
        run_env_worker=run_env_worker,
        run_policy_update_worker=run_policy_update_worker,
        make_env=make_env,
        make_algo=make_algo,
        make_policy=make_policy,
        make_replay_buffer=make_replay_buffer,
        make_logger=make_logger,
        config=config,
    )
