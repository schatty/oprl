import random
import threading
import numpy as np

from params import train_params
from utils.prioritised_experience_replay import PrioritizedReplayBuffer
from utils.gaussian_noise import GaussianNoiseGenerator
from learner import Learner
from agent import Agent


def train():
    # Prioritised experience replay memory
    per_memory = PrioritizedReplayBuffer(train_params.REPLAY_MEM_SIZE, train_params.PRIORITY_ALPHA)

    # Initialize Gaussian noise generator
    gaussian_noise = GaussianNoiseGenerator(train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.NOISE_SCALE)

    # Create threads for learner process and agent process
    threads = []

    # Create threading event for communication and synchronisation between the learner and agent threads
    run_agent_event = threading.Event()
    stop_agent_event = threading.Event()

    learner = Learner(per_memory)
    threads.append(threading.Thread(target=learner.run))

    for n_agent in range(1):
        print("N_AGENT: ", n_agent)
        # Initialize agent
        agent = Agent(env=train_params.ENV, actor_learner=learner.actor, n_agent=n_agent)
        print("Agent builded.")

        threads.append(threading.Thread(target=agent.run,
                                        args=(per_memory, gaussian_noise, run_agent_event, stop_agent_event)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    train()