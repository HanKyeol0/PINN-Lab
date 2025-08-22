import math
import numpy as np
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class Helmholtz2DSteady(BaseExperiment):
    """
    Steady 2D Helmholtz on (x,y) ∈ [xa,xb]×[ya,yb]:
        u_xx + u_yy + λ u = f(x,y)

    Manufactured solution:
        u*(x,y) = sin(a1 π x) sin(a2 π y)
      => u_xx + u_yy = - (a1^2 + a2^2) π^2 u*
      => choose f(x,y) = ( - (a1^2 + a2^2) π^2 + λ ) u*(x,y)

    We enforce Dirichlet BCs from u* on all 4 edges and fit the interior residual.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        ya, yb = cfg["domain"]["y"]
        self.rect = Rectangle(xa, xb, ya, yb, device)

        self.a1 = float(cfg.get("a1", 1.0))
        self.a2 = float(cfg.get("a2", 4.0))
        self.lam = float(cfg.get("lambda", 1.0))
        self.input_dim = 2  # (x,y)

    # ----- analytic fields -----
    def u_star(self, x, y):
        return torch.sin(self.a1 * math.pi * x) * torch.sin(self.a2 * math.pi * y)

    def f(self, x, y):
        coeff = (-(self.a1**2 + self.a2**2) * (math.pi**2) + self.lam)
        return coeff * self.u_star(x, y)

    # ----- sampling -----
    def sample_batch(self, n_f, n_b, n_0):
        # interior
        X_f = self.rect.sample(n_f)  # [n_f,2]

        # 4-edge Dirichlet BC from u*
        nb = max(1, n_b // 4)
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
        y = torch.rand(nb, 1, device=self.rect.device) * (yb - ya) + ya
        x = torch.rand(nb, 1, device=self.rect.device) * (xb - xa) + xa
        top    = torch.cat([x, torch.full_like(x, yb)], 1)
        bottom = torch.cat([x, torch.full_like(x, ya)], 1)
        left   = torch.cat([torch.full_like(y, xa), y], 1)
        right  = torch.cat([torch.full_like(y, xb), y], 1)
        X_b = torch.cat([top, bottom, left, right], dim=0)
        u_b = self.u_star(X_b[:, 0:1], X_b[:, 1:2])

        return {"X_f": X_f, "X_b": X_b, "u_b": u_b}  # no IC for steady case

    # ----- losses -----
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])          # [N,2] -> (x,y)
        u = model(X)                         # [N,1]
        du = grad_sum(u, X)                  # [N,2]
        u_x, u_y = du[:, 0:1], du[:, 1:2]
        d2ux = grad_sum(u_x, X)              # [N,2]
        d2uy = grad_sum(u_y, X)              # [N,2]
        u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]

        # residual_pred = u_xx + u_yy + λ u
        res_pred = u_xx + u_yy + self.lam * u

        # target forcing f(x,y)
        f_xy = self.f(X[:, 0:1], X[:, 1:2])
        return (res_pred - f_xy).pow(2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub).pow(2)

    def initial_loss(self, model, batch):
        # steady problem; no IC
        return torch.tensor(0.0, device=self.rect.device)

    # ----- eval & plots -----
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb,
                             nx, ny, self.rect.device)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx, ny)
            U_true = self.u_star(Xg, Yg)
        rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb,
                             nx, ny, self.rect.device)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx, ny).cpu().numpy()
            U_true = self.u_star(Xg, Yg).cpu().numpy()
        return save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred,
                             out_dir, "helmholtz2d_steady")
