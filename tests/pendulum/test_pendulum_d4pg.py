import unittest
from scripts.train import train, read_config


class TestsPendulumD4PG(unittest.TestCase):

    def test_d4pg_train(self):
        CONFIG_PATH = "tests/pendulum/config.yml"
        config = read_config(CONFIG_PATH)
        config['model'] = 'd4pg'
        config['num_agents'] = 1
        train(CONFIG_PATH, config=config)