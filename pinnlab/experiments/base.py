import torch, numpy as np, math
from abc import ABC, abstractmethod

def make_leaf(X: torch.Tensor) -> torch.Tensor:
    """Return a leaf tensor with requires_grad=True."""
    return X.clone().detach().requires_grad_(True)

def grad_sum(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx where y can be vector; uses sum-of-outputs trick."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

class BaseExperiment(ABC):
    def __init__(self, cfg, device):
        self.cfg = cfg; self.device = device

    @abstractmethod
    def sample_batch(self, n_f:int, n_b:int, n_0:int): ...

    def pde_residual_loss(self, model, batch): return torch.tensor(0., device=self.device)
    def boundary_loss(self, model, batch):     return torch.tensor(0., device=self.device)
    def initial_loss(self, model, batch):      return torch.tensor(0., device=self.device)
    def data_loss(self, model, batch):         return torch.tensor(0., device=self.device)

    @abstractmethod
    def relative_l2_on_grid(self, model, grid_cfg): ...

    @abstractmethod
    def plot_final(self, model, grid_cfg, out_dir): ...