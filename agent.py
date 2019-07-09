import os
import sys
import torch
import numpy as np
import scipy.stats as ss
from collections import deque
import cv2
import imageio

from params import train_params
from utils.network import Actor
from utils.env_wrapper import PendulumWrapper


class Agent:

    def __init__(self, env, actor_learner, n_agent=0):
        print(f"Initializing agent {n_agent}...")
        self.n_agent = n_agent

        # Create environment
        if env == "Pendulum-v0":
            self.env_wrapper = PendulumWrapper(env)
        else:
            raise Exception("Unknown environment")

        self.actor_learner = actor_learner
        self.actor = Actor(train_params.STATE_DIMS,
                               train_params.ACTION_DIMS,
                               train_params.ACTION_BOUND_LOW,
                               train_params.ACTION_BOUND_HIGH)

    def update_actor_learner(self):
        target = self.actor_learner
        source = self.actor
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def run(self, per_memory, gaussian_noise, run_agent_event, stop_agent_event):
        # Continuously run agent in environment to collect experiences and add to replay memory
        print("Agent run!")

        # Initialize deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        # Initially set threading event to allow agent to run until told otherwise
        run_agent_event.set()

        num_eps = 0
        while not stop_agent_event.is_set():
            num_eps += 1

            # Reset environment and experience buffer
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            self.exp_buffer.clear()

            num_steps = 0
            episode_reward = 0
            ep_done = False

            while not ep_done:
                num_steps += 1

                # Take action and store experience
                state = torch.tensor(state).float()
                #print("Before actor acts.")
                #print("State = ", type(state), state.shape)
                action = self.actor(state.unsqueeze(0))

                action = action.detach().numpy()

                action += (gaussian_noise() * train_params.NOISE_DECAY ** num_eps)
                #print("Action: ", type(action), action.shape)

                next_state, reward, terminal = self.env_wrapper.step(action)
                next_state = next_state.squeeze()
                #print("Next state: ", type(next_state), next_state.shape)

                episode_reward += reward

                next_state = self.env_wrapper.normalise_state(next_state)
                #print("Next state after normalisation: ", next_state.shape)
                reward = self.env_wrapper.normalise_reward(reward)

                self.exp_buffer.append((state, action, reward))

                # At least N steps in exp_buffer required to compute Bellman rewards
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE

                    # If learner is requesting a pause, wait before adding more samples
                    run_agent_event.wait()
                    #print("Adding to per memory!!!")
                    per_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)

                state = next_state

                if terminal or num_steps == train_params.MAX_EP_LENGTH:
                    # Compute Bellman rewards and add experiences to replay memory
                    # for the last N-1 experiences still remaining
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = train_params.DISCOUNT_RATE
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= train_params.DISCOUNT_RATE

                        # If learner is requesting a pause, wait before adding more samples
                        run_agent_event.wait()
                        #print("Adding final samples per memory!!!")
                        per_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)

                    # Start next episode
                    ep_done = True

                # Update agent networks with learner params every 'update_agent_ep' episodes
                if num_eps % train_params.UPDATE_AGENT_EP == 0:
                    #print("Performing hard update of the learner actor!")
                    self.update_actor_learner()

        self.env_wrapper.close()
