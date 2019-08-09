import shutil
import os
import time
from collections import deque
import matplotlib.pyplot as plt

from .utils import create_actor
from utils.utils import OUNoise, make_gif
from env.utils import create_env_wrapper
from utils.logger import Logger


class Agent(object):

    def __init__(self, config, actor_learner, global_episode, n_agent=0, log_dir=''):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.max_steps = config['max_ep_length']
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(self.env_wrapper.get_action_space())

        self.actor_learner = actor_learner
        self.actor = create_actor(model_name=config['model'],
                                  num_actions=config['action_dims'][0],
                                  num_states=config['state_dims'][0],
                                  hidden_size=config['dense_size'])
        # Logger
        log_path = f"{log_dir}/agent-{n_agent}.pkl"
        self.logger = Logger(log_path)

    def update_actor_learner(self):
        """Update local actor to the actor from learner. """
        source = self.actor_learner
        target = self.actor
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def run(self, replay_queue, stop_agent_event):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        rewards = []
        while not stop_agent_event.value:
            episode_reward = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.local_episode % 1 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")
            if self.global_episode.value >= self.config['num_episodes_train']:
                stop_agent_event.value = 1
                print("Stop agent!")
                break

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            for step in range(self.max_steps):
                action = self.actor.get_action(state)
                action = self.ou_noise.get_action(action, step)
                action = action.squeeze(0)
                next_state, reward, done = self.env_wrapper.step(action)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate']

                    if not stop_agent_event.value:
                        replay_queue.put([state_0, action_0, discounted_reward, next_state, done, gamma])

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Log metrics
            self.logger.scalar_summary("reward", episode_reward)
            self.logger.scalar_summary("episode_timing", time.time() - ep_start_time)

            rewards.append(episode_reward)
            if self.local_episode % self.config['update_agent_ep'] == 0:
                #print("Performing hard update of the local actor to the learner.")
                self.update_actor_learner()

        print("Emptying replay queue")
        while not replay_queue.empty():
            replay_queue.get()

        # Save replay from the first agent only
        if self.n_agent == 0:
            self.save_replay_gif()

        print(f"Agent {self.n_agent} done.")

    def save_replay_gif(self):
        dir_name = "replay_render"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = self.ou_noise.get_action(action, step)
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")