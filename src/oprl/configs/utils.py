import os
from datetime import datetime
import logging


def create_logdir(logdir: str, algo: str, env: str, seed: int):
    dt = datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
    log_dir = os.path.join(
        logdir, algo, 
        f"{algo}-env_{env}-seed_{seed}-{dt}")
    logging.info(f"LOGDIR: {log_dir}")
    return log_dir
