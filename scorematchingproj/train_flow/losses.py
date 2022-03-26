import torch
from functools import partial


def get_loss(loss_type, loss_init_kwargs):
    if loss_type == "dsm":
        return partial(dsm_loss, **loss_init_kwargs)
    elif loss_type == "deen":
        return partial(deen_loss, **loss_init_kwargs)
    elif loss_type == "ssm":
        return partial(ssm_loss, **loss_init_kwargs)
    elif loss_type == "ssm_vr":
        return partial(ssm_vr_loss, **loss_init_kwargs)
    else:
        raise ImportError


# `energy_model`: energy model, i.e. energy_model(x) = -log q(x), where q(x) is the unnormalized pdf


def dsm_loss(energy_model, x, sigma=0.1):
    """DSM loss from
    A Connection Between Score Matching and Denoising Autoencoders

    The loss is computed as
    x_ = x + v   # noisy samples
    s = -dE(x_)/dx_
    loss = 1/2*||s + (x-x_)/sigma^2||^2

    Args:
        x (torch.Tensor): input samples, (bsize, 2)
        sigma (float, optional): noise scale. Defaults to 0.1.

    Returns:
        DSM loss
    """
    x = x.requires_grad_()
    v = torch.randn_like(x) * sigma
    x_ = x + v
    s = energy_model.score(x_)
    loss = torch.norm(s + v / (sigma ** 2), dim=-1) ** 2
    loss = loss.mean() / 2.0
    return loss


# [CS]: I'm wondering that dsm and deen are equivalent
def deen_loss(energy_model, x, sigma=0.1):
    """DEEN loss from
    Deep Energy Estimator Networks

    The loss is computed as
    x_ = x + v   # noisy samples
    s = -dE(x_)/dx_
    loss = 1/2*||x - x_ + sigma^2*s||^2

    Args:
        x (torch.Tensor): input samples
        sigma (int, optional): noise scale. Defaults to 1.

    Returns:
        DEEN loss
    """
    x = x.requires_grad_()
    v = torch.randn_like(x) * sigma
    x_ = x + v
    s = sigma ** 2 * energy_model.score(x_)
    loss = torch.norm(s + v, dim=-1) ** 2
    loss = loss.mean() / 2.0
    return loss


def ssm_loss(energy_model, x, n_slices=1):
    """SSM loss from
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation

    The loss is computed as
    s = -dE(x)/dx
    loss = vT*(ds/dx)*v + 1/2*(vT*s)^2

    Args:
        x (torch.Tensor): input samples
        n_slices (torch.Tensor): number of slices, default=1

    Returns:
        SSM loss
    """
    x = x.unsqueeze(0).expand(n_slices, *x.shape)  # (n_slices, b, ...)
    x = x.contiguous().view(-1, *x.shape[2:])  # (n_slices*b, ...)
    x = x.requires_grad_()
    score = energy_model.score(x)  # (n_slices*b, ...)
    v = torch.randn((n_slices,) + x.shape, dtype=x.dtype, device=x.device)
    v = v.view(-1, *v.shape[2:])  # (n_slices*b, ...)
    sv = torch.sum(score * v)  # ()
    loss1 = torch.sum(score * v, dim=-1) ** 2 * 0.5  # (n_slices*b,)
    gsv = torch.autograd.grad(sv, x, create_graph=True)[0]  # (n_slices*b, ...)
    loss2 = torch.sum(v * gsv, dim=-1)  # (n_slices*b,)
    loss = (loss1 + loss2).mean()  # ()
    return loss


def ssm_vr_loss(energy_model, x, n_slices=1):
    """SSM-VR (variance reduction) loss from
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation

    The loss is computed as
    s = -dE(x)/dx
    loss = vT*(ds/dx)*v + 1/2*||s||^2

    Args:
        x (torch.Tensor): input samples
        n_slices (torch.Tensor): number of slices, default=1

    Returns:
        SSM-VR loss
    """
    x = x.unsqueeze(0).expand(n_slices, *x.shape)  # (n_slices, b, ...)
    x = x.contiguous().view(-1, *x.shape[2:])  # (n_slices*b, ...)
    x = x.requires_grad_()
    score = energy_model.score(x)  # (n_slices*b, ...)
    v = torch.randn((n_slices,) + x.shape, dtype=x.dtype, device=x.device)
    v = v.view(-1, *v.shape[2:])  # (n_slices*b, ...)
    sv = torch.sum(score * v)  # ()
    loss1 = torch.norm(score, dim=-1) ** 2 * 0.5  # (n_slices*b,)
    gsv = torch.autograd.grad(sv, x, create_graph=True)[0]  # (n_slices*b, ...)
    loss2 = torch.sum(v * gsv, dim=-1)  # (n_slices*b,)
    loss = (loss1 + loss2).mean()  # ()
    return loss
