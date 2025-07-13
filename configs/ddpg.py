from oprl.algos.protocols import AlgorithmProtocol
from oprl.algos.ddpg import DDPG
from oprl.buffers.protocols import ReplayBufferProtocol
from oprl.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.parse_args import parse_args
from oprl.logging import (
    LoggerProtocol,
    make_text_logger_func,
)
from oprl.environment import make_env as _make_env
from oprl.runners.train import run_training
from oprl.runners.config import CommonParameters


args = parse_args()

def make_env(seed: int):
    return _make_env(args.env, seed=seed)


env = make_env(seed=0)
STATE_DIM: int = env.observation_space.shape[0]
ACTION_DIM: int = env.action_space.shape[0]


config = CommonParameters(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    num_steps=int(100_000),
    eval_every=2500,
    device=args.device,
    estimate_q_every=5000,
    log_every=2500,
)


def make_algo(logger: LoggerProtocol) -> AlgorithmProtocol:
    return DDPG(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        logger=logger,
    ).create()


def make_replay_buffer() -> ReplayBufferProtocol:
    return EpisodicReplayBuffer(
        buffer_size_transitions=max(config.num_steps, int(1e6)),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=config.device,
    ).create()


make_logger = make_text_logger_func(
    algo="DDPG",
    env=args.env,
)


if __name__ == "__main__":
    run_training(
        make_algo=make_algo,
        make_env=make_env,
        make_replay_buffer=make_replay_buffer,
        make_logger=make_logger,
        config=config,
        seeds=args.seeds,
        start_seed=args.start_seed
    )


