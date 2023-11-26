from configs.base import TrainConfig


train = TrainConfig(
    env="walker-walk",
    algo="DDPG",
    log_dir="logs",
    num_steps=int(5_000_000),
    eval_every=2000,
    log_every=1000,
    seed=0,
    device="cpu",
    estimate_q_every=int(10_000),
    visualise_every=int(1_000_000),
    save_buffer=False,
)
