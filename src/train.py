import os
from datetime import datetime
import argparse

from trainers.base_trainer import BaseTrainer
from algos.ddpg import DDPG
from utils.logger import Logger
from env import DMControlEnv
print("Imports OK.")


def parse_args():

    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, help="Path to the config file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()


    ENV = "walker-walk"
    ALGO = "DDPG"
    LOGDIR = "logs"
    NUM_STEPS = 10000
    EVAL_INTERVAL = 2000
    LOG_EVERY=1000
    SEED = 0
    DEVICE = "cpu"
    ESTIMATE_Q_EVERY = 5000
    VISUALISE_EVERY = 10000
    SAVE_BUFFER = False

    config = dict(
        env=ENV,
        num_steps=int(100_000),
        algo=ALGO,
        device="cpu",
        seed=SEED,
    )

    env = DMControlEnv(ENV, seed=SEED)

    def make_test_env(seed: int):
        return DMControlEnv(ENV, seed)

    time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    log_dir = os.path.join(
        LOGDIR, ALGO, f"{ALGO}-env_{ENV}-seed{SEED}-{time}")
    logger = Logger(log_dir)
    print("LOGDIR: ", log_dir)

    
    STATE_SHAPE = env.observation_space.shape
    ACTION_SHAPE = env.action_space.shape

    trainer_class = BaseTrainer
    algo = DDPG(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device=DEVICE,
        seed=SEED,
        logger=logger
    )

    trainer = trainer_class(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        env=env,
        make_env_test=make_test_env,
        algo=algo,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL,
        device=DEVICE,
        save_buffer_every=SAVE_BUFFER,
        visualise_every=VISUALISE_EVERY,
        estimate_q_every=ESTIMATE_Q_EVERY,
        stdout_log_every=LOG_EVERY,
        seed=SEED,
        logger=logger,
    )

    '''
    config = read_config(args.config)
    trainer = config.make_trainer()

    '''

    print("Training starts...")
    trainer.train()
    print("OK.")
