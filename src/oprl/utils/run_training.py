import logging
from multiprocessing import Process

from oprl.trainers.base_trainer import BaseTrainer
from oprl.trainers.safe_trainer import SafeTrainer
from oprl.utils.utils import set_seed


def run_training(
    make_algo, make_env, make_logger, config, seeds: int = 1, start_seed: int = 0
):
    if seeds == 1:
        _run_training_func(make_algo, make_env, make_logger, config, 0)
    else:
        processes = []
        for seed in range(start_seed, start_seed + seeds):
            processes.append(
                Process(
                    target=_run_training_func,
                    args=(make_algo, make_env, make_logger, config, seed),
                )
            )

        for i, p in enumerate(processes):
            p.start()
            logging.info(f"Starting process {i}...")
        for p in processes:
            p.join()
        logging.info("Training finished.")


def _run_training_func(make_algo, make_env, make_logger, config, seed: int):
    set_seed(seed)
    env = make_env(seed=seed)
    logger = make_logger(seed)

    if env.env_family == "dm_control":
        trainer_class = BaseTrainer
    elif env.env_family == "safety_gymnasium":
        trainer_class = SafeTrainer
    else:
        raise ValueError(f"Unsupported env family: {env.env_family}")

    trainer = trainer_class(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        env=env,
        make_env_test=make_env,
        algo=make_algo(logger),
        num_steps=config["num_steps"],
        eval_interval=config["eval_every"],
        device=config["device"],
        estimate_q_every=config["estimate_q_every"],
        stdout_log_every=config["log_every"],
        seed=seed,
        logger=logger,
    )

    trainer.train()
