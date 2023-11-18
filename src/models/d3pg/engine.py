import copy
from datetime import datetime
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
from time import sleep
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import os

from models.agent import Agent
from utils.logger import Logger
from utils.utils import empty_torch_queue

from .d3pg import LearnerD3PG
from .networks import PolicyNetwork
from .utils import ReplayBuffer


def sampler_worker(config, replay_queue, batch_queue, training_on,
                   global_episode, update_step, log_dir=''):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.

    Args:
        config:
        replay_queue:
        batch_queue:
        training_on:
        global_episode:
        log_dir:
    """
    batch_size = config['batch_size']
    logger = Logger(f"{log_dir}/data_struct")

    # Create replay buffer
    replay_buffer = ReplayBuffer(state_dim=config["state_dim"],
                                 action_dim=config["action_dim"],
                                 max_size=config["replay_mem_size"],
                                 save_dir=config["results_path"])

    while training_on.value:
        # (1) Transfer replays to global buffer
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        try:
            batch = replay_buffer.sample(batch_size)
            batch_queue.put_nowait(batch)
        except:
            sleep(0.1)
            continue

        # Log data structures sizes
        step = update_step.value
        logger.scalar_summary("data_struct/global_episode", global_episode.value, step)
        logger.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
        logger.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
        logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def learner_worker(config, training_on, policy, target_policy_net, learner_w_queue,
                   batch_queue, update_step, experiment_dir):
    learner = LearnerD3PG(config, policy, target_policy_net, learner_w_queue, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, update_step)


def agent_worker(config, policy, learner_w_queue, global_episode, i, agent_type,
                 experiment_dir, training_on, replay_queue, update_step):
    agent = Agent(config,
                  policy=policy,
                  global_episode=global_episode,
                  n_agent=i,
                  agent_type=agent_type,
                  log_dir=experiment_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


class Engine(object):
    def __init__(self, config):
        self.config = config

    def train(self):
        config = self.config

        batch_queue_size = config['batch_queue_size']
        n_agents = config['num_agents']

        # Create directory for experiment
        experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=config["replay_queue_size"])
        training_on = mp.Value('i', 1)
        update_step = mp.Value('i', 0)
        global_episode = mp.Value('i', 0)
        learner_w_queue = torch_mp.Queue(maxsize=n_agents)

        # Data sampler
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        p = torch_mp.Process(target=sampler_worker,
                             args=(config, replay_queue, batch_queue, training_on,
                                   global_episode, update_step, experiment_dir))
        processes.append(p)

        # Learner (neural net training process)
        target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'],
                                          config['dense_size'], device=config['device'])
        policy_net = copy.deepcopy(target_policy_net)
        policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'],
                                          config['dense_size'], device=config['agent_device'])

        target_policy_net.share_memory()

        p = torch_mp.Process(target=learner_worker, args=(config, training_on, policy_net, target_policy_net, learner_w_queue,
                                                          batch_queue, update_step, experiment_dir))
        processes.append(p)

        # Single agent for exploitation
        p = torch_mp.Process(target=agent_worker,
                             args=(config, target_policy_net, None, global_episode, 0, "exploitation", experiment_dir,
                                   training_on, replay_queue, update_step))
        processes.append(p)

        # Agents (exploration processes)
        for i in range(n_agents):
            p = torch_mp.Process(target=agent_worker,
                                 args=(config, copy.deepcopy(policy_net_cpu), learner_w_queue, global_episode,
                                       i, "exploration", experiment_dir, training_on, replay_queue, update_step))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print("End.")

