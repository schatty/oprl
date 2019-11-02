import unittest
from models.engine import load_engine
from utils.utils import read_config


class TestsPendulum(unittest.TestCase):

    def test_ddpg(self):
        CONFIG_PATH = 'tests/config_test.yml'
        config = read_config(CONFIG_PATH)

        config['env'] = 'Pendulum-v0'
        config['model'] = 'ddpg'
        config['state_dim'] = 3
        config['action_dim'] = 1

        engine = load_engine(config)
        engine.train()

    def test_d4pg(self):
        CONFIG_PATH = 'tests/config_test.yml'
        config = read_config(CONFIG_PATH)

        config['env'] = 'Pendulum-v0'
        config['model'] = 'd4pg'
        config['state_dim'] = 3
        config['action_dim'] = 1

        engine = load_engine(config)
        engine.train()


