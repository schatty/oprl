import logging

from oprl.algos.td3 import TD3
from oprl.configs.utils import create_logdir, parse_args
from oprl.utils.utils import set_logging

set_logging(logging.INFO)
from oprl.env import make_env as _make_env
from oprl.utils.logger import FileLogger, Logger
from oprl.utils.run_training import run_training

args = parse_args()


def make_env(seed: int):
    return _make_env(args.env, seed=seed)


env = make_env(seed=0)
STATE_DIM: int = env.observation_space.shape[0]
ACTION_DIM: int = env.action_space.shape[0]


# --------  Config params -----------

config = {
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "num_steps": int(1_000_000),
    "eval_every": 2500,
    "device": args.device,
    "save_buffer": False,
    "visualise_every": 0,
    "estimate_q_every": 5000,
    "log_every": 2500,
}

# -----------------------------------


def make_algo(logger):
    return TD3(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        logger=logger,
    )


def make_logger(seed: int) -> Logger:
    global config

    log_dir = create_logdir(logdir="logs", algo="TD3", env=args.env, seed=seed)
    return FileLogger(log_dir, config)


if __name__ == "__main__":
    args = parse_args()
    run_training(make_algo, make_env, make_logger, config, args.seeds, args.start_seed)
