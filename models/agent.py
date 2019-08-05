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

    def __init__(self, config, actor_learner, n_agent=0):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.max_steps = train_params.MAX_EP_LENGTH

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
        state = self.env_wrapper.reset()
        self.ou_noise.reset()

        num_episode = 0
        rewards = []
        while not stop_agent_event.value:
            episode_reward = 0
            num_episode += 1
            if num_episode % 25 == 0:
                print("Episode: ", num_episode)
            if num_episode >= train_params.NUM_STEPS_TRAIN:
                stop_agent_event.value = 1
                print("Stop agent!")

            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            for step in range(self.max_steps):
                action = self.actor.get_action(state)
                action = self.ou_noise.get_action(action, step)
                next_state, reward, done, _ = self.env_wrapper.step(action)

                replay_queue.put((state, action, reward, next_state, done))

                state = next_state
                episode_reward += reward

                if done:
                    break

            rewards.append(episode_reward)
            if num_episode % train_params.UPDATE_AGENT_EP == 0:
                print("Performing hard update of the local actor to the learner.")
                self.update_actor_learner()
        print("Exit agent.")

        plot(num_episode, rewards)
        plt.savefig(f"reward_{self.config['model']}_{self.n_agent}.png")