import argparse
import os
from datetime import datetime
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--env", type=str, default="cartpole-balance", help="Name of the environment.")
    parser.add_argument("--seeds", type=int, default=1, help="Number of parallel processes launched with different random seeds.")
    parser.add_argument("--start_seed", type=int, default=0, help="Number of the first seed. Following seeds will be incremented from it.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to perform training on.")
    return parser.parse_args()


def create_logdir(logdir: str, algo: str, env: str, seed: int):
    dt = datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
    log_dir = os.path.join(
        logdir, algo, 
        f"{algo}-env_{env}-seed_{seed}-{dt}")
    logging.info(f"LOGDIR: {log_dir}")
    return log_dir
