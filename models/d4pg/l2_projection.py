## l2_projection ##
# Adapted for PyTorch from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py
# Projects the target distribution onto the support of the original network [Vmin, Vmax]

import torch


def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    print("z_p: ", z_p.shape)
    print("p: ", p.shape)
    print("z_q: ", z_q.shape)

    z_p = torch.tensor(z_p).float()

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]

    d_pos = torch.cat([z_q, vmin[None]], 0)[1:]
    d_neg = torch.cat([vmax[None], z_q], 0)[:-1]

    # Clip z_p to be in new support range (vmin, vmax)
    z_p = torch.clamp(z_p, vmin, vmax)[:, None, :]

    # Get the distance between atom values in support
    d_pos = (d_pos - z_q)[None, :, None]
    d_neg = (z_q - d_neg)[None, :, None]
    z_q = z_q[None, :, None]

    d_neg = torch.where(d_neg>0, 1./d_neg, torch.zeros(d_neg.shape))
    d_pos = torch.where(d_pos>0, 1./d_pos, torch.zeros(d_pos.shape))

    delta_qp = z_p - z_q
    d_sign = (delta_qp >= 0).type(p.dtype)

    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]
    return torch.sum(torch.clamp(1. - delta_hat, 0., 1.) * p, -1)
#
import numpy as np


def _l2_project2(next_distr_v, rewards_v, dones_mask_t, gamma, DELTA_Z, N_ATOMS, Vmin, Vmax, device="cpu"):
    #print("next_distr_v: ", next_distr_v.shape, type(next_distr_v))
    #print("rewards_v: ", rewards_v.shape, type(rewards_v))
    #print("dones_mask_t: ", dones_mask_t.shape, type(dones_mask_t))

    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    #print("DELTA_Z: ", DELTA_Z)
    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        #if atom == 0:
        #    print("tz_j: ", tz_j.shape)
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
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

    #print("proj_distr: ", proj_distr.shape, type(proj_distr))

    return torch.FloatTensor(proj_distr).to(device)