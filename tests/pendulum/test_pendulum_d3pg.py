import unittest
from models.d3pg.train import train, read_config


class TestsPendulumD3PG(unittest.TestCase):

    def test_d3pg_train(self):
        CONFIG_PATH = 'tests/pendulum/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        train(config)

    def test_d3pg_train_prioritized(self):
        CONFIG_PATH = 'tests/pendulum/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        config['replay_memory_prioritized'] = 1
        train(config)

