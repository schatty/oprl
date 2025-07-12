from oprl.algos import OffPolicyAlgorithm
from oprl.algos.ddpg import DDPG
from oprl.buffers.episodic_buffer import ReplayBufferProtocol, EpisodicReplayBuffer
from oprl.parse_args import parse_args
from oprl.logging import (
    set_logging,
    LoggerProtocol,
    make_text_logger_func,
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
    "visualise_every": 50000,
    "estimate_q_every": 5000,
    "gamma": 0.99,
    "log_every": 2500,
}

# -----------------------------------


def make_algo(logger: LoggerProtocol) -> OffPolicyAlgorithm:
    return DDPG(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        discount=config["gamma"],
        logger=logger,
    ).create()


def make_replay_buffer() -> ReplayBufferProtocol:
    return EpisodicReplayBuffer(
        buffer_size=max(config["num_steps"], int(1e6)),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=config["device"],
        gamma=config["gamma"],
    ).create()


make_logger = make_text_logger_func(
    config=config,
    algo="DDPG",
    env=args.env,
)


if __name__ == "__main__":
    args = parse_args()
    run_training(
        make_algo=make_algo,
        make_env=make_env,
        make_replay_buffer=make_replay_buffer,
        make_logger=make_logger,
        config=config,
        seeds=args.seeds,
        start_seed=args.start_seed
    )
