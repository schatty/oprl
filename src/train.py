import os
from datetime import datetime
import argparse

from trainers.base_trainer import BaseTrainer
from algos.ddpg import DDPG
from utils.logger import Logger
from utils.config import load_config
from env import DMControlEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str, help="Path to the config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)    
    print("TRAIN: ", config.train)

    env = DMControlEnv(config.train.env, seed=config.train.seed)

    def make_test_env(seed: int):
        return DMControlEnv(config.train.env, seed)

    time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    log_dir = os.path.join(
        config.train.log_dir, config.train.algo, 
        f"{config.train.algo}-env_{config.train.env}-seed{config.train.seed}-{time}")
    logger = Logger(log_dir)
    print("LOGDIR: ", log_dir)

    
    STATE_SHAPE = env.observation_space.shape
    ACTION_SHAPE = env.action_space.shape

    trainer_class = BaseTrainer
    algo = DDPG(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        device=config.train.device,
        seed=config.train.seed,
        logger=logger
    )

    trainer = trainer_class(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        env=env,
        make_env_test=make_test_env,
        algo=algo,
        num_steps=config.train.num_steps,
        eval_interval=config.train.eval_every,
        device=config.train.device,
        save_buffer_every=config.train.save_buffer,
        visualise_every=config.train.visualise_every,
        estimate_q_every=config.train.estimate_q_every,
        stdout_log_every=config.train.log_every,
        seed=config.train.seed,
        logger=logger,
    )

    print("Training starts...")
    trainer.train()
    print("OK.")
