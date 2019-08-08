import unittest
from scripts.train import train, read_config


class TestsPendulum(unittest.TestCase):

    def test_d3pg_train(self):
        CONFIG_PATH = 'tests/bipedal/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        config['num_agents'] = 2

        train(CONFIG_PATH, config=config)

    #def test_d4pg_train(self):
    #    config = read_config("tests/pendulum/config.yml")
    #    config['model'] = 'd4pg'

    #    train(config)