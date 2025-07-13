from pydantic_settings import BaseSettings


class CommonParameters(BaseSettings):
    state_dim: int
    action_dim: int
    num_steps: int
    eval_every: int = 2500
    estimate_q_every: int = 5000 
    log_every: int = 2500
    device: str = "cpu"
