import torch as t
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Update target network using Polyak-Ruppert Averaging."""
    with t.no_grad():
        for tgt, src in zip(target.parameters(), source.parameters()):
            tgt.data.mul_(1.0 - tau)
            tgt.data.add_(tau * src.data)


def disable_gradient(network: nn.Module) -> None:
    """Disable gradient calculations of the network."""
    for param in network.parameters():
        param.requires_grad = False
