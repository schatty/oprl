from oprl.trainers.base_trainer import BaseTrainer
from oprl.trainers.safe_trainer import SafeTrainer


def run_training(make_algo, make_env, make_logger, config, seed):
    env = make_env(seed=seed)
    logger = make_logger(seed)

    if env.env_family == "dm_control":
        trainer_class = BaseTrainer
    elif env.env_family == "safety_gymnasium":
        trainer_class = SafeTrainer
    else:
        raise ValueError(f"Unsupported env family: {env.env_family}")

    trainer = trainer_class(
        state_shape=config["state_shape"],
        action_shape=config["action_shape"],
        env=env,
        make_env_test=make_env,
        algo=make_algo(logger, seed),
        num_steps=config["num_steps"],
        eval_interval=config["eval_every"],
        device=config["device"],
        save_buffer_every=config["save_buffer"],
        visualise_every=config["visualise_every"],
        estimate_q_every=config["estimate_q_every"],
        stdout_log_every=config["log_every"],
        seed=seed,
        logger=logger,
    )

    trainer.train()
