import logging
import os
from datetime import datetime
import argparse
from multiprocessing import Process
from copy import copy, deepcopy

from oprl.trainers.base_trainer import run_training
from oprl.algos.distrib_ddpg import DistribDDPG
from oprl.utils.logger import Logger
from oprl.env import DMControlEnv
from oprl.configs.utils import create_logdir

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--env", type=str, default="cartpole-balance", help="Name of the environment.")
    parser.add_argument("--n_seed_processes", type=int, default=1, help="Number of parallel processes launched with different random seeds.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to perform training on.")
    return parser.parse_args()

args = parse_args()


def make_env(seed: int):
    return DMControlEnv(args.env, seed=seed)


env = DMControlEnv(args.env, seed=0)
STATE_SHAPE = env.observation_space.shape
ACTION_SHAPE = env.action_space.shape


# --------  Config params -----------

config = {
    "state_shape": STATE_SHAPE,
    "action_shape": ACTION_SHAPE,
    "num_steps": int(15_000),
    "eval_every": 2500,
    "device": args.device,
    "save_buffer": False,
    "visualise_every": 0,
    "estimate_q_every": 5000,
    "log_every": 1000
}

# -----------------------------------
 

def make_algo(logger, seed):
    return DistribDDPG(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device=args.device,
        seed=seed,
        logger=logger
    )


def make_logger(seed: int):
    log_dir = create_logdir(logdir="logs", algo="DistribDDPG", env=args.env, seed=seed)
    #TODO: add config here instead {}
    return Logger(log_dir, {})


if __name__ == "__main__":
    args = parse_args()

    if args.n_seed_processes == 1:
        run_training(make_algo, make_env, make_logger, config, 0)
    else:
        processes = []
        for seed in range(args.n_seed_processes):
            processes.append(
                    Process(target=run_training, args=(make_trainer, seed))
            )

        for i, p in enumerate(processes):
            p.start()
            print(f"Starting process {i}...")

        for p in processes:
            p.join()

    print("Training end.")
