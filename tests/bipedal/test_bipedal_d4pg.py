import unittest
from models.d3pg.train import train, read_config


class TestsBipedalWalkerD4PG(unittest.TestCase):

    def test_d4pg_train(self):
        CONFIG_PATH = 'tests/bipedal/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd4pg'
        train(config=config)