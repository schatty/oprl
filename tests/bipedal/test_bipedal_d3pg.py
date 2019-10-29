import unittest
from models.d3pg.train import train, read_config


class TestsBipedalWalkerD3PG(unittest.TestCase):

    def test_d3pg_train(self):
        CONFIG_PATH = 'tests/bipedal/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        train(config=config)
