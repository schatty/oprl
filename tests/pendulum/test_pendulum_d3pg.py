import unittest
from scripts.train import train, read_config


class TestsPendulumD3PG(unittest.TestCase):

    def test_d3pg_train_multiprocessing(self):
        CONFIG_PATH = 'tests/pendulum/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        train(config)

