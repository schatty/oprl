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

            # ------------- Critic training step -----------

            # Predict actions for next  states by passing next states through policy target network
            next_states_batch = torch.tensor(next_states_batch)
            future_actions = self.actor_target(next_states_batch)

            # Predict future Z distribution by passing next states and actions through value target network
            target_Z_dist = self.critic_target(next_states_batch, future_actions)
            target_Z_atoms = self.critic_target.z_atoms

            # Create batch of target network's Z-atoms
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), train_params.BATCH_SIZE, axis=0) # [batch_size x n_atoms]

            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals_batch, :] = 0.0

            gammas_batch = gammas_batch.reshape((-1, 1)) # [batch_size x 1]

            target_Z_atoms = rewards_batch + (target_Z_atoms * gammas_batch)

            self.z_atoms = torch.linspace(train_params.V_MIN,
                                          train_params.V_MAX,
                                          train_params.NUM_ATOMS)

            criterion = nn.BCELoss(reduction='none')
            optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

            target_Z_projected = _l2_project(target_Z_atoms,
                                             target_Z_dist,
                                             self.z_atoms)

            t = torch.autograd.Variable(target_Z_projected, requires_grad=False)

            states_batch = torch.tensor(states_batch).float()
            actions_batch = torch.tensor(actions_batch).float()
            actions_batch = actions_batch.squeeze(-1)
            critic_out = self.critic(states_batch, actions_batch)
            loss = criterion(critic_out, t)

            TD_error = torch.sum(loss, 1) # [batch_size x 1]
            TD_error *= torch.tensor(weights_batch).float()

            loss_final = torch.mean(loss)

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            # Use critic TD errors to update sample priorities
            td_error_np = TD_error.detach().numpy()
            self.per_memory.update_priorities(idx_batch, (np.abs(td_error_np) + train_params.PRIORITY_EPSILON))

            # ------------- Actor training step ------------

            # Get policy network's action outputs for selected states
            states_batch = torch.tensor(states_batch)
            actor_actions = self.actor(states_batch)

            # Gradient of mean of output Z-distribution wrt action input
            # used to train actor network, weighing the grads by z_values
            # given the mean across the output distribution
            critic_output = self.critic(states_batch, actor_actions)
            action_grads = torch.zeros(critic_output.shape)
            torch.autograd.grad(critic_output, actor_actions, action_grads)
            action_grads *= self.z_atoms

            # Train actor
            actions_pred = self.actor(states_batch)
            actor_loss = -self.critic(states_batch, actions_pred).mean()

            # Compute gradients of critic's value output distribution wrt actions
            optimizer_actor = torch.optim.Adam(self.actor.parameters(), 0.001)
            optimizer_actor.zero_grad()
            actor_loss.backward()

            # Update target networks
            self.soft_update(self.critic, self.critic_target, train_params.TAU)
            self.soft_update(self.actor, self.actor_target, train_params.TAU)

            # Increment beta value at end of every step
            priority_beta += beta_increment

            # Periodically check capacity of replay memory and remove old samples
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.per_memory) > train_params.REPLAY_MEM_SIZE:
                    samples_to_remove = len(self.per_memory) - train_params.REPLAY_MEM_SIZE
                    self.per_memory.remove(samples_to_remove)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        theta_target = tau * theta_local * (1 - tau) * theta_target

        Params:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
