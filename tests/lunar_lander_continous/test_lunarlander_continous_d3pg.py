import unittest
from models.d3pg.train import train, read_config


class TestsLunarLanderContinousD3PG(unittest.TestCase):

    def test_d3pg_train(self):
        CONFIG_PATH = 'tests/lunar_lander_continous/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        train(CONFIG_PATH)

    def test_d3pg_train_prioritized(self):
        CONFIG_PATH = 'tests/lunar_lander_continous/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        config['replay_memory_prioritized'] = 1
        train(CONFIG_PATH)