import random
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from models.utils import create_learner
from models.agent import Agent
from params import train_params


def sampler_worker(replay_queue, batch_queue, stop_agent_event, batch_size):
    replay_buffer = []
    while not stop_agent_event.value or not replay_queue.empty():
        if not replay_queue.empty():
            replay = replay_queue.get()
            replay_buffer.append(replay)

        if len(replay_buffer) < batch_size:
            continue

        idxes = [random.randint(0, len(replay_buffer) - 1) for _ in range(batch_size)]
        elems = [replay_buffer[i] for i in idxes]
        batch_queue.put(elems)

    print("Stop sampler worker.")


def train():
    config = {'model': 'd4pg', 'environment': 'Pendulum-v0'}
    batch_size = train_params.BATCH_SIZE

    processes = []
    replay_queue = torch_mp.Queue(maxsize=64)
    stop_agent_event = torch_mp.Value('i', 0)
    global_episode = torch_mp.Value('i', 0)
    n_agents = 2

    batch_queue = torch_mp.Queue(maxsize=64)
    p = torch_mp.Process(target=sampler_worker, args=(replay_queue, batch_queue, stop_agent_event, batch_size))
    processes.append(p)

    learner = create_learner(config, batch_queue)
    p = torch_mp.Process(target=learner.run, args=(stop_agent_event,))
    processes.append(p)

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
    train()
