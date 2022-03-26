import os
import sys
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import tqdm

from scorematchingproj.helper.score_based_sampling import langevin_dynamics, anneal_langevin_dynamics


def plot_data(ax, data, range_lim=4, bins=1000, cmap=plt.cm.viridis):
    rng = [[-range_lim, range_lim], [-range_lim, range_lim]]
    ax.hist2d(data[:, 0], data[:, 1], range=rng, bins=bins, cmap=cmap)


def plot_scores(ax, mesh, scores, width=0.002):
    """Plot score field

    Args:
        ax (): canvas
        mesh (np.ndarray): mesh grid
        scores (np.ndarray): scores
        width (float, optional): vector width. Defaults to 0.002
    """
    ax.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=width)


def plot_energy(ax, energy, cmap=plt.cm.viridis, flip_y=True):
    if flip_y:
        energy = energy[::-1]  # flip y
    ax.imshow(energy, cmap=cmap)


def plot_samples(ax, score_fn, langevin_steps, eps, device="cpu"):
    samples = []
    for i in tqdm.tqdm(range(50)):
        x = torch.rand(20000, 2) * 8 - 4
        x = x.to(device=device)
        x = langevin_dynamics(score_fn, x, n_steps=langevin_steps, eps=eps).detach().cpu().numpy()
        samples.append(x)
    samples = np.concatenate(samples, axis=0)
    # draw energy
    ax.grid(False)
    ax.axis("off")
    plot_data(ax, samples)
    ax.set_title("Sampled data", fontsize=16)


def sample_score_field(score_fn, range_lim=4, grid_size=50, device="cpu"):
    """Sampling score field from an energy model

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        range_lim (int, optional): Range of x, y coordimates. Defaults to 4.
        grid_size (int, optional): Grid size. Defaults to 50.
        device (str, optional): torch device. Defaults to 'cpu'.
    """
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    mesh = np.transpose(np.asarray(np.meshgrid(x, y)).reshape(2, -1), (1, 0))
    x = torch.from_numpy(mesh).float()
    x = x.to(device=device)
    scores = score_fn(x.detach()).detach()
    scores = scores.cpu().numpy()
    return mesh, scores


def plot_score_field(ax, score_fn, device="cpu"):
    mesh, scores = sample_score_field(score_fn, device=device)
    # draw scores
    ax.grid(False)
    ax.axis("off")
    plot_scores(ax, mesh, scores)
    ax.set_title("Estimated scores", fontsize=16)


def sample_energy_field(energy_fn, range_lim=4, grid_size=1000, split_num=100, device="cpu"):
    """Sampling energy field from an energy model

    Args:
        energy_fn (callable): an energy function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        range_lim (int, optional): range of x, y coordinates. Defaults to 4.
        grid_size (int, optional): grid size. Defaults to 1000.
        split_num: avoid putting all mesh into "gpu", split more if out of memory
        device (str, optional): torch device. Defaults to 'cpu'.
    """
    energy = []
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    mesh = np.transpose(np.asarray(np.meshgrid(x, y)).reshape(2, -1), (1, 0))
    for mesh_chunk in np.split(mesh, split_num):
        inputs = torch.from_numpy(mesh_chunk).float().to(device=device)
        energy.append(energy_fn(inputs.detach()).detach().cpu().numpy())
    energy = np.stack(energy, axis=0).reshape(grid_size, grid_size)  # (grid_size, grid_size)
    return energy


def plot_energy_field(ax, model, device="cpu"):
    energy = sample_energy_field(model, device=device)
    # draw energy
    ax.grid(False)
    ax.axis("off")
    plot_energy(ax, energy)
    ax.set_title("Estimated energy", fontsize=16)
