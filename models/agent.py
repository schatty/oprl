from collections import deque
import gym
import matplotlib.pyplot as plt

from params import train_params
from utils.env_wrapper import PendulumWrapper
from .utils import create_actor
from utils.utils import OUNoise, NormalizedActions, ReplayBuffer


def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f"frame {frame_idx}. reward {rewards[-1]}")
    plt.plot(rewards)


class Agent:

    def __init__(self, config, actor_learner, global_episode, n_agent=0):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.max_steps = train_params.MAX_EP_LENGTH
        self.global_episode = global_episode
        self.local_episode = 0

        # Create environment
        env = config["environment"]
        if env == "Pendulum-v0":
            self.env_wrapper = PendulumWrapper(env)
        else:
            raise Exception("Unknown environment")

        self.env_wrapper = NormalizedActions(gym.make("Pendulum-v0"))
        self.ou_noise = OUNoise(self.env_wrapper.action_space)

        self.actor_learner = actor_learner
        self.actor = create_actor(model_name=config['model'], num_actions=train_params.ACTION_DIMS[0], num_inputs=train_params.STATE_DIMS[0],
                                  hidden_size=train_params.DENSE1_SIZE)

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
            if self.global_episode.value >= train_params.NUM_STEPS_TRAIN:
                stop_agent_event.value = 1
                print("Stop agent!")

            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            for step in range(self.max_steps):
                action = self.actor.get_action(state)
                action = self.ou_noise.get_action(action, step)
                next_state, reward, done, _ = self.env_wrapper.step(action)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE

                    replay_queue.put((state_0, action_0, discounted_reward, next_state, done))

                #replay_queue.put((state, action, reward, next_state, done))

                state = next_state
                episode_reward += reward

                if done:
                    break

            rewards.append(episode_reward)
            if self.local_episode % train_params.UPDATE_AGENT_EP == 0:
                print("Performing hard update of the local actor to the learner.")
                self.update_actor_learner()
        print("Exit agent.")

        plot(self.local_episode, rewards)
        plt.savefig(f"reward_{self.config['model']}_{self.n_agent}.png")