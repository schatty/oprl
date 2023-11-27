from copy import deepcopy
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


from oprl.algos.nn import Critic, MLP
from oprl.algos.utils import initialize_weight


# WIP here


## Code of projection was taken from:
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter14/06_train_d4pg.py
def _l2_project(next_distr_v, rewards_v, dones_mask_t, gamma, delta_z, n_atoms, v_min, v_max):
    print("next_distr_v", next_distr_v.shape)
    print("rewards_v", rewards_v.shape)
    print("dones_mask_t", dones_mask_t.shape)
    print("delta_z", delta_z.shape)

    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(bool)
    print("dones_mask shape: ", dones_mask.shape)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)

    for atom in range(n_atoms):
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards + (v_min + atom * delta_z) * gamma))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        print("proj_distr: ", proj_distr.shape)
        print("eq_mask: ", eq_mask.shape)
        print("l u: ", l.shape, u.shape)
        print("l[eq_mask]: ", l[eq_mask].shape)
        print("next_dist: ", next_distr.shape)
        print("atom", atom)
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones_mask]))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

    return proj_distr


class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, state_shape, action_shape, hidden_size, v_min, v_max,
                 num_atoms, device='cuda'):
        super().__init__()

        self.linear1 = nn.Linear(state_shape[0] + action_shape[0], hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_atoms)

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_probs(self, state, action):
        return torch.softmax(self.forward(state, action), dim=1)


class DeterministicPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True), device: str="cpu"):
        super().__init__()

        self.mlp = MLP(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).apply(initialize_weight)

        self._device = device
        self._action_shape = action_shape

    def forward(self, states):
        return torch.tanh(self.mlp(states))

    def exploit(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self._device)
        return self.forward(state).cpu().data.numpy().flatten()

    def explore(self, state: npt.ArrayLike) -> npt.ArrayLike:
        state = torch.tensor(
            state, dtype=torch.float, device=self._device).unsqueeze_(0)

        # TODO: Set exploration noise of 0.1 as the property of the agent
        with torch.no_grad():
            noise = (torch.randn(self._action_shape) * 0.1).to(self._device)
            action = self.mlp(state) + noise

        a = action.cpu().numpy()[0]
        # TODO: Set the action limit in the init
        return np.clip(a, -1.0, 1.0)


class DistribDDPG:
    def __init__(self, state_shape, action_shape, max_action=1, discount=0.99, tau=5e-3,
                 batch_size=256, device="cpu", seed=0, logger=None):
        np.random.seed(seed)
        torch.manual_seed(seed)


        self.logger = logger

        #TODO: add n-step return
        self.expl_noise = 0.1
        self.num_atoms = 51
        self.v_min = 0
        self.v_max = 63.4
        self.gamma = 0.99
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.action_shape = action_shape
        self.dtype = torch.float
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device

        self.actor = DeterministicPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = ValueNetwork(state_shape, action_shape, 256, self.v_min, self.v_max, self.num_atoms).to(self.device)
        # self.critic = Critic(state_shape, action_shape).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        # TODO: maybe simplify setting of the separate loss
        self.critic_losss = nn.BCELoss(reduction='none')

    def exploit(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # TODO: remove explore from algo to agent completely
    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)

        with torch.no_grad():
            noise = (torch.randn(self.action_shape) * self.max_action * self.expl_noise).to(self.device)
            action = self.actor(state) + noise

        a = action.cpu().numpy()[0]
        return np.clip(a, -self.max_action, self.max_action)

    def update(self, batch):
        # Sample replay buffer 
        #state, action, next_state, reward, done = batch
        state, action, reward, done, next_state = batch

        # Compute the target Q value
        target_val = self.critic_target.get_probs(next_state, self.actor_target(next_state))
        print("target val: ", target_val.shape)

        # Get projected distribution
        target_z_projected = _l2_project(next_distr_v=target_val,
                                         rewards_v=reward,
                                         dones_mask_t=done,
                                         gamma=self.gamma,
                                         n_atoms=self.num_atoms,
                                         v_min=self.v_min,
                                         v_max=self.v_max,
                                         delta_z=self.delta_z)
        target_z_projected = torch.from_numpy(target_z_projected).float().to(self.device)
        print("target_z_projected: ", target_z_projected)

        # Get current Q estimate
        current_val = self.critic.get_probs(state, action)
        current_val = current_val.to(self.device)
        print("Current val: ", current_val.shape)

        # Compute critic loss
        critic_loss = self.critic_loss(current_Q, target_Q).mean(axis=1)
        print("critic loss: ", critic_loss.shape)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        print("Critic update ok.")

        # Compute actor loss
        actor_loss = self.critic.get_probs(state, self.actor(state))
        # TODO can I do this with tensors right away?
        actor_loss *= torch.from_numpy(self.critic.z_atoms).float().to(self.device)
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        print("Actor update ok.")

        _ = input("stop update step")

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 

    def get_policy_state_dict(self) -> OrderedDict:
        return self.actor.state_dict()
