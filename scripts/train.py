from time import sleep
from datetime import datetime
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import os
from shutil import copyfile

from models.utils import create_learner
from models.agent import Agent
from utils.utils import read_config
from utils.logger import Logger
from utils.prioritized_experience_replay import create_replay_buffer


def sampler_worker(config, replay_queue, batch_queue, replay_priorities_queue, training_on,
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
    num_agents = config['num_agents']
    batch_size = config['batch_size']
    priority_beta_start = config['priority_beta_start']
    num_steps_train = config['num_steps_train']
    priority_beta_increment = (config['priority_beta_end'] - config['priority_beta_start']) / num_steps_train

    # Logger
    fn = f"{log_dir}/data_struct.pkl"
    logger = Logger(log_path=fn)

    # Create replay buffer
    replay_buffer = create_replay_buffer(config)

    while training_on.value or not replay_queue.empty():
        # (1) Transfer replays to global buffer
        #for _ in range(num_agents):
        #    if not replay_queue.empty():
        #        replay = replay_queue.get()
        #        replay_buffer.add(*replay)
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if not training_on.value:
            # Repeat loop to wait until replay_queue will be empty
            continue
        if len(replay_buffer) < batch_size:
            continue

        if not replay_priorities_queue.empty():
            inds, weights = replay_priorities_queue.get()
            replay_buffer.update_priorities(inds, weights)

        priority_beta = priority_beta_start + (update_step.value + 1) * priority_beta_increment
        batch = replay_buffer.sample(batch_size, beta=priority_beta)
        batch_queue.put(batch)

        # Log data structures sizes
        logger.scalar_summary("global_episode", global_episode.value)
        logger.scalar_summary("replay_queue", replay_queue.qsize())
        logger.scalar_summary("batch_queue", batch_queue.qsize())
        logger.scalar_summary("replay_buffer", len(replay_buffer))


    while not replay_priorities_queue.empty():
        replay_priorities_queue.get()
    print("Stop sampler worker.")


def train(config):
    config_path = None
    if isinstance(config, str):
        config_path = config
        config = read_config(config)
    elif not isinstance(config, dict):
        raise ValueError("config should be either string (path to config) or dict.")

    replay_queue_size = config['replay_queue_size']
    batch_queue_size = config['batch_queue_size']
    n_agents = config['num_agents']

    # Create directory for experiment
    experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, f"{experiment_dir}/config.yml")

    # Data structures
    processes = []
    replay_queue = torch_mp.Queue(maxsize=replay_queue_size)
    replay_priorities_queue = torch_mp.Queue(maxsize=batch_queue_size)
    training_on = torch_mp.Value('i', 1)
    update_step = torch_mp.Value('i', 0)
    global_episode = torch_mp.Value('i', 0)

    # Data sampler
    batch_queue = torch_mp.Queue(maxsize=batch_queue_size)
    p = torch_mp.Process(target=sampler_worker,
                         args=(config, replay_queue, batch_queue, replay_priorities_queue, training_on,
                               global_episode, update_step, experiment_dir))
    processes.append(p)

    # Learner (neural net training process)
    learner = create_learner(config, log_dir=experiment_dir)
    p = torch_mp.Process(target=learner.run, args=(training_on, batch_queue, replay_priorities_queue, update_step))
    processes.append(p)

    # Agents (exploration processes)
    for i in range(n_agents):
        agent = Agent(config,
                      actor_learner=learner.target_policy_net,
                      global_episode=global_episode,
                      n_agent=i,
                      log_dir=experiment_dir)
        p = torch_mp.Process(target=agent.run, args=(training_on, replay_queue, update_step))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("End.")


if __name__ == "__main__":
    CONFIG_PATH = "config.yml"
    train(CONFIG_PATH)
