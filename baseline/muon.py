"""Muon optimizer implementation with Newton-Schulz orthogonalization.

Reference: Jordan et al., 2024
"""

import torch
from torch.optim.optimizer import Optimizer


def newton_schulz_iter(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalization.

    Iteratively computes: G_ortho = (3*I - G^T @ G) @ G / 2

    Args:
        G: Input gradient tensor (*, d_in, d_out)
        steps: Number of iterations

    Returns:
        Orthogonalized gradient tensor
    """
    if G.ndim < 2:
        return G

    # Normalize to prevent numerical issues
    scale = G.norm()
    G = G / (scale + 1e-8)

    for _ in range(steps):
        G = (3 * G - G @ (G.T @ G)) / 2

    return G * scale


class Muon(Optimizer):
    """Muon optimizer for dense neural network weights.

    Applies Newton-Schulz orthogonalization to Nesterov momentum gradients.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum coefficient (default: 0.9)
        ns_steps: Newton-Schulz iteration steps (default: 5)
    """

    def __init__(self, params, lr: float, momentum: float = 0.9, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # Nesterov momentum
                buf.mul_(momentum).add_(grad)
                G_momentum = grad.add(buf, alpha=momentum)

                # Newton-Schulz orthogonalization for 2D+ tensors
                if G_momentum.ndim >= 2:
                    G_ortho = newton_schulz_iter(G_momentum, steps=ns_steps)
                else:
                    G_ortho = G_momentum

                # Update parameters
                p.add_(G_ortho, alpha=-lr)

        return loss
