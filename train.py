import argparse
from models.engine import load_engine
from utils.utils import read_config

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, help="Path to the config file.")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    config = read_config(args['config'])
    engine = load_engine(config)
    engine.train()