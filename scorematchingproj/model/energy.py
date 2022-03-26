from pathlib import Path
import numpy as np
import torch
from torch import nn

# --- primitives ---
class Swish(nn.Module):
    def __init__(self, dim=-1):
        """Swish activ bootleg from
        https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299

        Args:
            dim (int, optional): input/output dimension. Defaults to -1.
        """
        super().__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)


class ToyMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, units=[300, 300], swish=True, dropout=False):
        """Toy MLP from
        https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py#L198
        Args:
            input_dim (int, optional): input dimensions. Defaults to 2.
            output_dim (int, optional): output dimensions. Defaults to 1.
            units (list, optional): hidden units. Defaults to [300, 300].
            swish (bool, optional): use swish as activation function. Set False to use
                soft plus instead. Defaults to True.
            dropout (bool, optional): use dropout layers. Defaults to False.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    Swish(out_dim) if swish else nn.Softplus(),
                    nn.Dropout(0.5) if dropout else nn.Identity(),
                ]
            )
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==================== Energy Model ====================
# E(x) = -log q(x), where q(x) is the unnormalized pdf, i.e. p(x) = q(x) / Z, where Z is partition
class Energy(nn.Module):
    def __init__(self, net):
        """A simple energy model

        Args:
            net (nn.Module): An energy function, the output shape of
                the energy function should be (b, 1). The score is
                computed by grad(-E(x))
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logq = -self.net(x).sum()
        return torch.autograd.grad(logq, x, create_graph=True)[0]

    def save(self, path):
        par_path = Path(path).parent
        par_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self
