from models.engine import load_engine
from utils.utils import read_config


if __name__ == "__main__":
    CONFIG_PATH = "configs/lunar_lander_contiuous_ddpg.yml"
    config = read_config(CONFIG_PATH)

    engine = load_engine(config)
    engine.train()