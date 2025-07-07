from oprl.algos import OffPolicyAlgorithm
from oprl.algos.sac import SAC
from oprl.parse_args import parse_args
from oprl.logging import (
    create_logdir,
    set_logging,
    FileTxtLogger,
    LoggerProtocol
)
set_logging()
from oprl.environment import make_env as _make_env
from oprl.train import run_training

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
    "num_steps": int(100_000),
    "eval_every": 2500,
    "device": args.device,
    "visualise_every": 0,
    "estimate_q_every": 5000,
    "log_every": 1000,
}

# -----------------------------------


def make_algo(logger: LoggerProtocol) -> OffPolicyAlgorithm:
    return SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        logger=logger,
    ).create()


def make_logger(seed: int) -> LoggerProtocol:
    log_dir = create_logdir(logdir="logs", algo="SAC", env=args.env, seed=seed)
    logger = FileTxtLogger(log_dir, config)
    logger.copy_source_code()
    return logger


if __name__ == "__main__":
    args = parse_args()
    run_training(make_algo, make_env, make_logger, config, args.seeds, args.start_seed)
