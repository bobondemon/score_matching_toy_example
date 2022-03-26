import torch
import numpy as np


def langevin_dynamics(score_fn, x, eps=0.1, n_steps=1000):
    """Langevin dynamics
    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        x (torch.Tensor): input samples
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
    """
    for i in range(n_steps):
        x = x + eps / 2.0 * score_fn(x).detach()
        x = x + torch.randn_like(x) * np.sqrt(eps)
    return x


def anneal_langevin_dynamics(score_fn, x, sigmas=None, eps=0.1, n_steps_each=100):
    """Annealed Langevin dynamics

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor, sigma: float) -> torch.Tensor
        x (torch.Tensor): input samples
        sigmas (torch.Tensor, optional): noise schedule. Defualts to None.
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
    """
    # default sigma schedule
    if sigmas is None:
        sigmas = np.exp(np.linspace(np.log(20), 0.0, 10))

    for sigma in sigmas:
        for i in range(n_steps_each):
            cur_eps = eps * (sigma / sigmas[-1]) ** 2
            x = x + cur_eps / 2.0 * score_fn(x, sigma).detach()
            x = x + torch.randn_like(x) * np.sqrt(eps)
    return x
