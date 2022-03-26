import time
import logging
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from scorematchingproj.train_flow.losses import get_loss
from scorematchingproj.helper.vis import (
    plot_data,
    plot_scores,
    plot_energy,
    plot_samples,
    plot_score_field,
    plot_energy_field,
)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        loss_type,
        loss_init_kwargs=None,
        vis_dir="./vis_dir",
        langevin_steps=100,
        langevin_eps=0.01,
        learning_rate=1e-3,
        clipnorm=100.0,
        device="cuda",
        tb_logdir="./tblog",
    ):
        """Energy based model trainer
        Args:
            model (nn.Module): energy-based model
            train_loader (ToyDataLoader): training dataloader
            loss_type: contains loss_type, one of ('dsm', 'deen', 'ssm', and 'ssm_vr')
            loss_init_kwargs: init kwargs for loss function
            vis_dir: visualization dir, default="./vis_dir"
            langevin_steps: step number for langevin dynamic sampling, default=100
            langevin_eps: noise eps for langevin dynamic sampling, default=0.01
            learning_rate (float, optional): learning rate. Defaults to 1e-4.
            clipnorm (float, optional): gradient clip. Defaults to 100..
            device (str, optional): torch device. Defaults to 'cuda'.
            tb_logdir: tensorboard log dir, if None, will be './tblog'
        """
        self.model = model
        self.train_loader = train_loader
        self.loss = get_loss(loss_type, loss_init_kwargs)
        self.data = np.asarray(self.train_loader.get_data())
        self.vis_dir = vis_dir
        self.langevin_steps = langevin_steps
        self.langevin_eps = langevin_eps
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.device = device

        self.model = self.model.to(device=self.device)
        # setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_gradsteps = 0
        self.num_steps = 0
        self.progress = 0
        self.tb_writer = SummaryWriter(tb_logdir)

    def train_step(self, batch, update=True):
        """Train one batch
        Args:
            batch: batch data (bsize, 2)
            update (bool, optional): whether to update networks.
                Defaults to True.
        Returns:
            loss
        """
        # move inputs to device
        x = torch.tensor(batch, dtype=torch.float32, device=self.device)
        # compute loss
        loss = self.loss(self.model, x)
        # update model
        if update:
            # compute gradients
            loss.backward()
            # perform gradient updates
            total_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def visualize(self, vis_dir, vis_name, data, langevin_steps=100, eps=0.01):
        logging.info("Visualizing data ...")
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / vis_name
        fig, axs = plt.subplots(figsize=(24, 6), ncols=4)
        # draw data samples
        axs[0].grid(False)
        axs[0].axis("off")
        plot_data(axs[0], data)
        axs[0].set_title("Ground truth data", fontsize=16)
        plot_samples(axs[1], self.model.score, self.langevin_steps, self.langevin_eps, device=self.device)
        plot_energy_field(axs[2], self.model, device=self.device)
        plot_score_field(axs[3], self.model.score, device=self.device)
        for ax in axs:
            ax.set_box_aspect(1)
        plt.tight_layout()
        fig.savefig(vis_path, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close("all")

    def learn(self, n_steps=50000, batch_size=100, log_freq=2500, vis_freq=2500):
        """Train the model
        Args:
            n_steps (int, optional): number of training steps. Defaults to 50000
            batch_size (int, optional): batch size. Defaults to 100
            log_freq (int, optional): logging frequency (step). Defaults to 2500
            vis_freq (int, optional): visualizing frequency (step). Defaults to 2500
        Returns:
            None
        """
        # initialize
        time_start = time.time()
        time_spent = 0

        all_losses = []
        for step_idx, batch in zip(range(n_steps), iter(self.train_loader)):
            loss = self.train_step(batch)
            all_losses.append(loss)
            if step_idx % log_freq == 0:
                # write tensorboard
                avgloss = torch.mean(torch.as_tensor(all_losses, dtype=torch.float64))
                self.tb_writer.add_scalar(f"train/loss", avgloss, step_idx)
                logging.info(f"[Step {step_idx}/{n_steps}]: avgloss: {avgloss}")
                all_losses = []  # clear out losses

            if step_idx % vis_freq == 0:
                logging.debug("Visualizing")
                self.model.eval()
                self.visualize(
                    self.vis_dir,
                    f"{step_idx}.png",
                    self.data,
                    self.langevin_steps,
                    self.langevin_eps,
                )
                self.model.train()

        logging.debug("Visualizing Last Time")
        self.model.eval()
        self.visualize(
            self.vis_dir,
            f"{n_steps}.png",
            self.data,
            self.langevin_steps,
            self.langevin_eps,
        )
        self.model.train()
