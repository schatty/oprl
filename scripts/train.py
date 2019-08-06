import random
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import gym
import yaml

from models.utils import create_learner
from models.agent import Agent


def read_config(path):
    """
    Return python dict from .yml file.

    Args:
        path (str): path to the .yml config.

    Returns (dict): configuration object.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Load environment from gym to set its params
    print("env: ", cfg['env'])
    env = gym.make(cfg['env'])
    cfg['state_dims'] = env.observation_space.shape
    cfg['state_bound_low'] = env.observation_space.low
    cfg['state_bound_high'] = env.observation_space.high
    cfg['action_dims'] = env.action_space.shape
    cfg['action_bound_low'] = env.action_space.low
    cfg['action_bound_high'] = env.action_space.high
    del env

    return cfg


def sampler_worker(config, replay_queue, batch_queue, stop_agent_event):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.

    Args:
        config:
        replay_queue:
        batch_queue:
        stop_agent_event:

    Returns:

    """
    num_agents = config['num_agents']
    batch_size = config['batch_size']

    # TODO: Replace with data structure
    replay_buffer = []

    while not stop_agent_event.value or not replay_queue.empty():
        # (1) Transfer replays to global buffer
        for _ in range(num_agents):
            if replay_queue.empty():
                break
            replay = replay_queue.get()
            replay_buffer.append(replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue
        idxes = [random.randint(0, len(replay_buffer) - 1) for _ in range(batch_size)]
        elems = [replay_buffer[i] for i in idxes]
        batch_queue.put(elems)

    print("Stop sampler worker.")


def train(config):

    # Config
    replay_queue_size = config['replay_queue_size']
    batch_queue_size = config['batch_queue_size']
    n_agents = config['num_agents']

    # Data structures
    processes = []
    replay_queue = torch_mp.Queue(maxsize=replay_queue_size)
    stop_agent_event = torch_mp.Value('i', 0)
    global_episode = torch_mp.Value('i', 0)

    # Data sampler
    batch_queue = torch_mp.Queue(maxsize=batch_queue_size)
    p = torch_mp.Process(target=sampler_worker, args=(config, replay_queue, batch_queue, stop_agent_event))
    processes.append(p)

    # Learner (neural net training process)
    learner = create_learner(config, batch_queue)
    p = torch_mp.Process(target=learner.run, args=(stop_agent_event,))
    processes.append(p)

    # Agents (exploration processes)
    for i in range(n_agents):
        agent = Agent(config, actor_learner=learner.target_policy_net, global_episode=global_episode, n_agent=i)
        p = torch_mp.Process(target=agent.run, args=(replay_queue, stop_agent_event))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("End.")


if __name__ == "__main__":
    config = read_config("config.yml")
    train(config)
