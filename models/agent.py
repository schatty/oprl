import tempfile
import os
from collections import deque
import matplotlib.pyplot as plt

from .utils import create_actor
from utils.utils import OUNoise, ReplayBuffer, make_gif
from env.utils import create_env_wrapper


def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f"frame {frame_idx}. reward {rewards[-1]}")
    plt.plot(rewards)


class Agent(object):

    def __init__(self, config, actor_learner, global_episode, n_agent=0):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.max_steps = config['max_ep_length']
        self.global_episode = global_episode
        self.local_episode = 0

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(self.env_wrapper.get_action_space())

        self.actor_learner = actor_learner
        self.actor = create_actor(model_name=config['model'],
                                  num_actions=config['action_dims'][0],
                                  num_states=config['state_dims'][0],
                                  hidden_size=config['dense_size'])

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

            if self.local_episode % 25 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")
            if self.global_episode.value >= self.config['num_episodes_train']:
                stop_agent_event.value = 1
                self.save_replay_gif()
                print("Stop agent!")
                break

            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            for step in range(self.max_steps):
                action = self.actor.get_action(state)
                action = self.ou_noise.get_action(action, step)
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

                    replay_queue.put((state_0, action_0, discounted_reward, next_state, done))

                state = next_state
                episode_reward += reward

                if done:
                    break

            rewards.append(episode_reward)
            if self.local_episode % self.config['update_agent_ep'] == 0:
                print("Performing hard update of the local actor to the learner.")
                self.update_actor_learner()
        print("Exit agent.")

        plot(self.local_episode, rewards)
        output_dir = self.config['results_path']
        plt.savefig(f"{output_dir}/reward-{self.config['model']}-process_{self.n_agent}-{episode_reward}.png")

    def save_replay_gif(self):
        output_dir = self.config['results_path']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with tempfile.TemporaryDirectory() as tmpdirname:
            state = self.env_wrapper.reset()
            for step in range(self.max_steps):
                action = self.actor.get_action(state)
                action = self.ou_noise.get_action(action, step)
                next_state, reward, done = self.env_wrapper.step(action)
                img = self.env_wrapper.render()
                plt.imsave(fname=f"{tmpdirname}/{step}.png", arr=img)
                state = next_state
                if done:
                    break

            fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
            make_gif(tmpdirname, f"{output_dir}/{fn}")
        print("fig saved to ", f"{output_dir}/{fn}")