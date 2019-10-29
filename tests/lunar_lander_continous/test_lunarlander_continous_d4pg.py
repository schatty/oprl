import unittest
from models.d3pg.train import train, read_config


class TestsLunarLanderContinousD4PG(unittest.TestCase):

    def test_d4pg_train(self):
        CONFIG_PATH = 'tests/lunar_lander_continous/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd4pg'
        train(config)

    def test_d4pg_train_prioritized(self):
        CONFIG_PATH = 'tests/lunar_lander_continous/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd4pg'
        config['replay_memory_prioritized'] = 1
        train(config)