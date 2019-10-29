import unittest
from models.d3pg.train import train, read_config


class TestsPendulumD4PG(unittest.TestCase):

    def test_d4pg_train(self):
        CONFIG_PATH = "tests/pendulum/config.yml"
        config = read_config(CONFIG_PATH)
        config['model'] = 'd4pg'
        train(config=config)

    def test_d4pg_train_prioritised(self):
        CONFIG_PATH = "tests/pendulum/config.yml"
        config = read_config(CONFIG_PATH)
        config['model'] = 'd4pg'
        config['replay_memory_prioritized'] = 1
        train(config=config)