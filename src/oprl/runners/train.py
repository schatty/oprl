from typing import Callable
import logging
import random
from multiprocessing import Process

import numpy as np
import torch as t

from oprl.algos.protocols import AlgorithmProtocol
from oprl.buffers.protocols import ReplayBufferProtocol
from oprl.environment.protocols import EnvProtocol
from oprl.trainers.base_trainer import BaseTrainer
from oprl.trainers.safe_trainer import SafeTrainer
from oprl.logging import LoggerProtocol
from oprl.runners.config import CommonParameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def run_training(
    make_algo: Callable[[LoggerProtocol], AlgorithmProtocol],
    make_env: Callable[[int], EnvProtocol],
    make_replay_buffer: Callable[[], ReplayBufferProtocol],
    make_logger: Callable[[int], LoggerProtocol],
    config: CommonParameters,
    seeds: int = 1,
    start_seed: int = 0
) -> None:
    if seeds == 1:
        _run_training_func(make_algo, make_env, make_replay_buffer, make_logger, config, 0)
    else:
        processes = []
        for seed in range(start_seed, start_seed + seeds):
            processes.append(
                Process(
                    target=_run_training_func,
                    args=(make_algo, make_env, make_replay_buffer, make_logger, config, seed),
                )
            )

        for i, p in enumerate(processes):
            p.start()
            logging.info(f"Starting process {i}...")
        for p in processes:
            p.join()
        logging.info("Training finished.")


def _run_training_func(
        make_algo: Callable[[LoggerProtocol], AlgorithmProtocol],
        make_env: Callable[[int], EnvProtocol],
        make_replay_buffer: Callable[[], ReplayBufferProtocol],
        make_logger: Callable[[int], LoggerProtocol],
        config: CommonParameters,
        seed: int,
) -> None:
    set_seed(seed)
    env = make_env(seed=seed)
    replay_buffer = make_replay_buffer()
    logger = make_logger(seed)
    algo = make_algo(logger)

    base_trainer = BaseTrainer(
        env=env,
        make_env_test=make_env,
        algo=algo,
        replay_buffer=replay_buffer,
        num_steps=config.num_steps,
        eval_interval=config.eval_every,
        device=config.device,
        estimate_q_every=config.estimate_q_every,
        stdout_log_every=config.log_every,
        seed=seed,
        logger=logger,
    )
    if env.env_family == "dm_control":
        trainer = base_trainer
    elif env.env_family == "safety_gymnasium":
        trainer = SafeTrainer(trainer=base_trainer)
    else:
        raise ValueError(f"Unsupported env family: {env.env_family}")
    trainer.train()
