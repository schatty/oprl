from pydantic_settings import BaseSettings


class TrainConfig(BaseSettings):
    env: str
    algo: str
    log_dir: str = "log_dir"
    num_steps: int = 100000
    seed: int = 0
    device: str = "cpu"
    log_every: int = 10000
    eval_every: int = 2000
    estimate_q_every: int = 5000
    visualise_every: int = 5000
    save_buffer: bool = False
