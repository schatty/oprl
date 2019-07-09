import os
import sys
import torch
import torch.nn as nn
import numpy as np
from time import sleep

from params import train_params
from utils.network import Actor, Critic
from utils.l2_projection import _l2_project


class Learner:
    def __init__(self, per_memory):
        self.per_memory = per_memory

        self.critic = Critic(train_params.STATE_DIMS,
                             train_params.ACTION_DIMS,
                             train_params.NUM_ATOMS,
                             train_params.V_MIN,
                             train_params.V_MAX)
        self.critic_target = Critic(train_params.STATE_DIMS,
                                    train_params.ACTION_DIMS,
                                    train_params.NUM_ATOMS,
                                    train_params.V_MIN,
                                    train_params.V_MAX)

        self.actor = Actor(train_params.STATE_DIMS,
                           train_params.ACTION_DIMS,
                           train_params.V_MIN,
                           train_params.V_MAX)
        self.actor_target = Actor(train_params.STATE_DIMS,
                                  train_params.ACTION_DIMS,
                                  train_params.V_MIN,
                                  train_params.V_MAX)

    def run(self):
        """
        Sample batches of experiences from replay memory and train learner networks
        """
        print("Learner Run!")

        # Initialize beta to start value
        priority_beta = train_params.PRIORITY_BETA_START
        beta_increment = (train_params.PRIORITY_BETA_END - train_params.PRIORITY_BETA_START) / train_params.NUM_STEPS_TRAIN

        # Can only train when we have at least batch_size num of samples replay memory
        while len(self.per_memory) <= train_params.BATCH_SIZE:
            print("Populating replay memory up to batch_size samples...", len(self.per_memory))
            sleep(1)
        else:
            print("Done")

        for train_step in range(1, train_params.NUM_STEPS_TRAIN+1):
            print("TRAIN STEP : ", train_step)
            # Get minibatch
            minibatch = self.per_memory.sample(train_params.BATCH_SIZE, priority_beta)

            states_batch = minibatch[0]
            actions_batch = minibatch[1]
            rewards_batch = minibatch[2]  # [batch_size x 1]
            next_states_batch = minibatch[3]
            terminals_batch = minibatch[4]
            gammas_batch = minibatch[5]
            weights_batch = minibatch[6]
            idx_batch = minibatch[7]

            print("States: ", states_batch.shape)
            print("Actions: ", actions_batch.shape)
            print("Rewards; ", rewards_batch.shape)
            print("Next state batch: ", next_states_batch.shape)
            print("Terminals batch: ", terminals_batch.shape)
            print("Gammas batch: ", gammas_batch.shape)
            print("Weights batch: ", weights_batch.shape)
            print("idx_batch: ", idx_batch)

            # ------------- Critic training step -----------

            print("Critic training step")
            # Predict actions for next  states by passing next states through policy target network
            next_states_batch = torch.tensor(next_states_batch)
            print("next_states_batch: ", type(next_states_batch), next_states_batch.shape)
            future_actions = self.actor_target(next_states_batch)
            print("future_actions: ", type(future_actions), future_actions.shape)

            # Predict future Z distribution by passing next states and actions through value target network
            target_Z_dist = self.critic_target(next_states_batch, future_actions)
            target_Z_atoms = self.critic_target.z_atoms

            # Create batch of target network's Z-atoms
            #target_Z_atoms = target_Z_atoms.numpy().reshape((-1, 1))
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), train_params.BATCH_SIZE, axis=0) # [batch_size x n_atoms]

            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals_batch, :] = 0.0

            gammas_batch = gammas_batch.reshape((-1, 1)) # [batch_size x 1]

            print("expanded rewards: ", rewards_batch.shape)
            print("target_Z_atoms: ", target_Z_atoms.shape)
            print("gammas expanded: ", gammas_batch.shape)

            target_Z_atoms = rewards_batch + (target_Z_atoms * gammas_batch)
            print("target_Z_atoms final: ", target_Z_atoms.shape)

            self.z_atoms = torch.linspace(train_params.V_MIN,
                                          train_params.V_MAX,
                                          train_params.NUM_ATOMS)

            criterion = nn.BCELoss()#nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

            print("Types: ", type(target_Z_atoms), type(target_Z_dist), type(self.z_atoms))

            target_Z_projected = _l2_project(target_Z_atoms,
                                             target_Z_dist,
                                             self.z_atoms)

            t = torch.autograd.Variable(target_Z_projected, requires_grad=False)

            loss = criterion(target_Z_dist, t)#target_Z_projected)
            #loss = criterion(target_Z_projected, target_Z_dist)

            print("loss: ", loss.shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # td_error [batch_size x 1]

            # ------------- Actor training step ------------

            # Increment beta value at end of every step
            priority_beta += beta_increment

            # Periodically check capacity of replay memory and remove old samples
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.per_memory) > train_params.REPLAY_MEM_SIZE:
                    samples_to_remove = len(self.per_memory) - train_params.REPLAY_MEM_SIZE
                    self.per_memory.remove(samples_to_remove)

            break

def soft_update(target, source, tau):
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        target_p.data.copy_(target_p.data * (1.0 - tau)  + source_p.data * tau)

