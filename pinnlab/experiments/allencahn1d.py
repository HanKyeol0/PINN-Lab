import math
import torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Interval

class AllenCahn1D(BaseExperiment):
    r"""
    Allen–Cahn (1D, time-dependent) with manufactured forcing:

        u_t - ε^2 u_xx + (u^3 - u) = f(x,t),   x∈[xa,xb], t∈[t0,t1]

    We choose the analytic (manufactured) solution:
        u*(x,t) = sin(π x) · cos(ω t)

    Then we set
        f(x,t) = u*_t - ε^2 u*_{xx} + (u*^3 - u*)
    and use Dirichlet BC/IC from u* so the residual can be driven to ≈0.

    cfg:
      domain: {x: [xa, xb], t: [t0, t1]}
      eps: 0.01         # ε
      omega: 2.0        # ω in u*
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        self.t0, self.t1 = cfg["domain"]["t"]
        self.x_dom = Interval(xa, xb, device)
        self.eps   = float(cfg.get("eps", 0.01))
        self.omega = float(cfg.get("omega", 2.0))

    # ----- manufactured truth -----
    def u_star(self, x, t):
        return torch.sin(math.pi * x) * torch.cos(self.omega * t)

    def f(self, x, t):
        u  = self.u_star(x, t)
        ut = -self.omega * torch.sin(math.pi * x) * torch.sin(self.omega * t)
        u_xx = -(math.pi**2) * self.u_star(x, t)
        return ut - (self.eps**2) * u_xx + (u**3 - u)

    # ----- batching -----
    def sample_batch(self, n_f: int, n_b: int, n_0: int):
        # interior (x,t)
        x_f = self.x_dom.sample(n_f)
        t_f = torch.rand(n_f, 1, device=self.x_dom.device) * (self.t1 - self.t0) + self.t0
        X_f = torch.cat([x_f, t_f], dim=1)

        # boundary at x=xa and x=xb (Dirichlet from u*)
        nb2 = max(1, n_b // 2)
        t_b = torch.rand(nb2, 1, device=self.x_dom.device) * (self.t1 - self.t0) + self.t0
        Xb_L = torch.cat([torch.full_like(t_b, self.x_dom.a), t_b], dim=1)
        Xb_R = torch.cat([torch.full_like(t_b, self.x_dom.b), t_b], dim=1)
        X_b  = torch.cat([Xb_L, Xb_R], dim=0)
        u_b  = self.u_star(X_b[:, 0:1], X_b[:, 1:2])

        # initial condition at t=t0
        x0 = self.x_dom.sample(n_0)
        X_0 = torch.cat([x0, torch.full_like(x0, self.t0)], dim=1)
        u0  = self.u_star(X_0[:, 0:1], X_0[:, 1:2])

        return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}

    # ----- losses -----
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])       # [N,2] = (x,t)
        x, t = X[:, 0:1], X[:, 1:2]
        u    = model(X)
        du   = grad_sum(u, X)             # [N,2] -> (u_x, u_t)
        u_x, u_t = du[:, 0:1], du[:, 1:2]
        d2u_x = grad_sum(u_x, X)          # [N,2] -> (u_xx, u_xt)
        u_xx  = d2u_x[:, 0:1]

        res = u_t - (self.eps**2) * u_xx + (u**3 - u) - self.f(x, t)
        return res.pow(2)

    def boundary_loss(self, model, batch):
        pred = model(batch["X_b"])
        return (pred - batch["u_b"]).pow(2)

    def initial_loss(self, model, batch):
        pred = model(batch["X_0"])
        return (pred - batch["u0"]).pow(2)

    # ----- eval / plots -----
    def relative_l2_on_grid(self, model, grid_cfg):
        import numpy as np
        nx, nt = grid_cfg["nx"], grid_cfg["nt"]
        xs = torch.linspace(self.x_dom.a, self.x_dom.b, nx, device=self.x_dom.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.x_dom.device)
        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT   = torch.stack([X.reshape(-1), T.reshape(-1)], 1)
        with torch.no_grad():
            U_pred = model(XT).reshape(nx, nt)
        U_true = self.u_star(X, T)
        rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_1d
        nx, nt = grid_cfg["nx"], grid_cfg["nt"]
        xs = torch.linspace(self.x_dom.a, self.x_dom.b, nx, device=self.x_dom.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.x_dom.device)
        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT   = torch.stack([X.reshape(-1), T.reshape(-1)], 1)
        with torch.no_grad():
            U_pred = model(XT).reshape(nx, nt).cpu().numpy()
        U_true = self.u_star(X, T).cpu().numpy()
        return save_plots_1d(xs.cpu().numpy(), ts.cpu().numpy(), U_true, U_pred, out_dir, "allencahn1d")
