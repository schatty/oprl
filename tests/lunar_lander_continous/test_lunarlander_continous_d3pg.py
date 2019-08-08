import unittest
from scripts.train import train, read_config


class TestsLunarLanderContinous(unittest.TestCase):

    def test_d3pg_train(self):
        CONFIG_PATH = 'tests/lunar_lander_continous/config.yml'
        config = read_config(CONFIG_PATH)
        config['model'] = 'd3pg'
        config['num_agents'] = 2

        train(CONFIG_PATH, config=config)