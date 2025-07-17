from pydantic_settings import BaseSettings


class CommonParameters(BaseSettings):
    state_dim: int
    action_dim: int
    num_steps: int
    eval_every: int = 2500
    estimate_q_every: int = 5000 
    log_every: int = 2500
    device: str = "cpu"


class DistribConfig(BaseSettings):
    batch_size: int = 128
    num_env_workers: int = 4
    episodes_per_worker: int = 100
    warmup_epochs: int = 16
    episode_length: int = 1000
    learner_num_waits: int = 10
    warmup_env_steps: int = 1000
    
