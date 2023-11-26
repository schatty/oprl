import torch
import torch.nn as nn


class Clamp(nn.Module):

    def forward(self, log_stds):
        return log_stds.clamp_(-20, 2)


def initialize_weight(m, gain=nn.init.calculate_gain('relu')):
    # Initialize linear layers with the orthogonal initialization.
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        m.bias.data.fill_(0.0)

    # Initialize conv layers with the delta-orthogonal initialization.
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def soft_update(target, source, tau):
    """ Update target network using Polyak-Ruppert Averaging. """
    with torch.no_grad():
        for tgt, src in zip(target.parameters(), source.parameters()):
            tgt.data.mul_(1.0 - tau)
            tgt.data.add_(tau * src.data)


def disable_gradient(network):
    """ Disable gradient calculations of the network. """
    for param in network.parameters():
        param.requires_grad = False
