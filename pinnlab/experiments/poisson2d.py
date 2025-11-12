import math
import numpy as np
import torch

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class Poisson2D(BaseExperiment):
    """
    Unified steady/time-dependent Poisson:

    - Steady Poisson:
        -∇² u = f(x,y)
        Choose u*(x,y) = sin(pi x) sin(pi y)  ->  ∇²u* = -2π² u*
        => f = 2π² u*, BC: Dirichlet from u*.

    - Time-dependent Heat:
        u_t - κ ∇²u = f(x,y,t)
        Choose u*(x,y,t) = sin(pi x) sin(pi y) exp(-λ t)
        ∇²u* = -2π² u*, u_t = -λ u*
        residual = (-λ + 2 κ π²) u* - f
        If λ = 2 κ π²  =>  f ≡ 0 (homogeneous). BC and IC from u*.

    Config keys:
      time_dependent: bool
      domain.x/y[/t]
      kappa, lambda ('auto') if time_dependent
      in_features: 2 (steady) or 3 (time-dependent)
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.time_dep = bool(cfg.get("time_dependent", False))

        xa, xb = cfg["domain"]["x"]
        ya, yb = cfg["domain"]["y"]
        self.rect = Rectangle(xa, xb, ya, yb, device)

        if self.time_dep:
            self.t0, self.t1 = cfg["domain"]["t"]
            self.kappa = float(cfg.get("kappa", 1.0))
            lam_cfg = cfg.get("lambda", "auto")
            if isinstance(lam_cfg, str) and lam_cfg.lower() == "auto":
                self.lmbda = 2.0 * self.kappa * (math.pi ** 2)
            else:
                self.lmbda = float(lam_cfg)
            self.input_dim = 3
        else:
            self.input_dim = 2
            
        self.sampling_mode = cfg.get("sampling_mode", "random") # random / grid
        if self.sampling_mode == "grid":
            g = cfg.get("grid", {})
            nx, ny = int(g.get("nx", 100)), int(g.get("ny", 100))
            
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, device)
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [nx*ny, 2]
            self._XY = XY
            
            if self.time_dep:
                nt = int(g.get("nt", 100))
                self._T = torch.linspace(self.t0, self.t1, nt, device=device) # [nt]
            
            # boundary subset (any edge)
            xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
            mask_b = XY[:,0].eq(xa) | XY[:,0].eq(xb) | XY[:,1].eq(ya) | XY[:,1].eq(yb)
            self._XYb = XY[mask_b]  # [nb, 2]

    # ---------- analytic fields ----------
    def u_star_steady(self, x, y):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y)

    def f_steady(self, x, y):
        # -∇²u* = 2π² u*
        return 2.0 * (math.pi ** 2) * self.u_star_steady(x, y)

    def u_star_time(self, x, y, t):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.exp(-self.lmbda * t)

    def f_time(self, x, y, t):
        # residual = (-λ + 2 κ π²) u* - f  => choose f accordingly (0 if λ = 2κπ²)
        coeff = (-self.lmbda + 2.0 * self.kappa * (math.pi ** 2))
        return coeff * self.u_star_time(x, y, t)

    # ---------- sampling ----------
    def sample_batch(self, n_f, n_b, n_0):
        if self.sampling_mode == "random":
            if self.time_dep:
                # Interior (x,y,t)
                X_f_xy = self.rect.sample(n_f)
                t_f = torch.rand(n_f, 1, device=self.rect.device) * (self.t1 - self.t0) + self.t0
                X_f = torch.cat([X_f_xy, t_f], dim=1)

                # Boundary over space at random t
                xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
                t_b = torch.rand(4 * n_b, 1, device=self.rect.device) * (self.t1 - self.t0) + self.t0

                y = torch.rand(n_b, 1, device=self.rect.device) * (yb - ya) + ya
                x = torch.rand(n_b, 1, device=self.rect.device) * (xb - xa) + xa
                top    = torch.cat([x, torch.full_like(x, yb)], 1)
                bottom = torch.cat([x, torch.full_like(x, ya)], 1)
                left   = torch.cat([torch.full_like(y, xa), y], 1)
                right  = torch.cat([torch.full_like(y, xb), y], 1)
                X_b_spatial = torch.cat([top, bottom, left, right], dim=0)
                X_b = torch.cat([X_b_spatial, t_b], dim=1)
                u_b = self.u_star_time(X_b[:, 0:1], X_b[:, 1:2], X_b[:, 2:3])

                # Initial condition at t = t0
                x0y0 = self.rect.sample(n_0)
                t0 = torch.full((n_0, 1), self.t0, device=self.rect.device)
                X_0 = torch.cat([x0y0, t0], dim=1)
                u0 = self.u_star_time(X_0[:, 0:1], X_0[:, 1:2], X_0[:, 2:3])

                return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}
            else:
                # Steady: interior (x,y)
                X_f = self.rect.sample(n_f)
                # Boundary at all 4 edges (Dirichlet from u*)
                xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
                y = torch.rand(n_b, 1, device=self.rect.device) * (yb - ya) + ya
                x = torch.rand(n_b, 1, device=self.rect.device) * (xb - xa) + xa
                top    = torch.cat([x, torch.full_like(x, yb)], 1)
                bottom = torch.cat([x, torch.full_like(x, ya)], 1)
                left   = torch.cat([torch.full_like(y, xa), y], 1)
                right  = torch.cat([torch.full_like(y, xb), y], 1)
                X_b = torch.cat([top, bottom, left, right], dim=0)
                u_b = self.u_star_steady(X_b[:, 0:1], X_b[:, 1:2])
                return {"X_f": X_f, "X_b": X_b, "u_b": u_b}
        elif self.sampling_mode == "grid":
            if self.time_dep:
                M, T = self._XY.size(0), self._T.size(0)

                # interior
                idx_xy = torch.randint(0, M, (n_f,), device=self.device)
                idx_t  = torch.randint(0, T, (n_f,), device=self.device)
                X_f = torch.cat([self._XY[idx_xy], self._T[idx_t].unsqueeze(1)], dim=1)
                
                # boundary (any edge)
                mb = self._XYb.size(0)
                idx_b = torch.randint(0, mb, (n_b,), device=self.device)
                idx_bt = torch.randint(0, T, (n_b,), device=self.device)
                X_b = torch.cat([self._XYb[idx_b], self._T[idx_bt].unsqueeze(1)], dim=1)
                u_b = self.u_star(X_b[:,0:1], X_b[:,1:2], X_b[:,2:3])

                # initial (t = t0)
                idx0 = torch.randint(0, M, (n_0,), device=self.device)
                X_0 = torch.cat([self._XY[idx0], torch.full((n_0,1), self.t0, device=self.device)], dim=1)
                u0 = self.u_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])

                return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}
            else:
                M = self._XY.size(0)

                # interior
                idx_xy = torch.randint(0, M, (n_f,), device=self.device)
                X_f = self._XY[idx_xy]

                # boundary (any edge)
                mb = self._XYb.size(0)
                idx_b = torch.randint(0, mb, (n_b,), device=self.device)
                X_b = self._XYb[idx_b]
                u_b = self.u_star_steady(X_b[:,0:1], X_b[:,1:2])

                return {"X_f": X_f, "X_b": X_b, "u_b": u_b}

    # ---------- losses ----------
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])
        u = model(X)

        if self.time_dep:
            # X: [N,3] = (x,y,t)
            du = grad_sum(u, X)              # [N,3]
            u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
            d2ux = grad_sum(u_x, X)
            d2uy = grad_sum(u_y, X)
            u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
            x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
            res = u_t - self.kappa * (u_xx + u_yy) - self.f_time(x, y, t)
            return res.pow(2)
        else:
            # X: [N,2] = (x,y)
            du = grad_sum(u, X)              # [N,2]
            u_x, u_y = du[:, 0:1], du[:, 1:2]
            d2ux = grad_sum(u_x, X)
            d2uy = grad_sum(u_y, X)
            u_xx, u_yy = d2ux[:, 0:1], d2uy[:, 1:2]
            x, y = X[:, 0:1], X[:, 1:2]
            res = -(u_xx + u_yy) - self.f_steady(x, y)
            return res.pow(2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub).pow(2)

    def initial_loss(self, model, batch):
        if not self.time_dep or ("X_0" not in batch):
            return torch.tensor(0.0, device=self.device)
        X0, u0 = batch["X_0"], batch["u0"]
        pred = model(X0)
        return (pred - u0).pow(2)

    # ---------- evaluation ----------
    def relative_l2_on_grid(self, model, grid_cfg):
        if self.time_dep:
            nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
            ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
            idxs = [0, nt // 2, nt - 1] if nt >= 3 else list(range(nt))
            rels = []
            with torch.no_grad():
                for ti in idxs:
                    T = torch.full_like(Xg, ts[ti])
                    XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                    U_pred = model(XYT).reshape(nx, ny)
                    U_true = self.u_star_time(Xg, Yg, T)
                    rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
                    rels.append(rel.item())
            return float(np.mean(rels))
        else:
            nx, ny = grid_cfg["nx"], grid_cfg["ny"]
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
            with torch.no_grad():
                U_pred = model(XY).reshape(nx, ny)
                U_true = self.u_star_steady(Xg, Yg)
            rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
            return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        if self.time_dep:
            nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
            ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
            figs = {}
            with torch.no_grad():
                for label, ti in zip(["t0", "tmid", "t1"], [0, nt // 2, nt - 1]):
                    T = torch.full_like(Xg, ts[ti])
                    XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                    U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                    U_true = self.u_star_time(Xg, Yg, T).cpu().numpy()
                    out = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"poisson2d_time_{label}")
                    figs.update(out)
            return figs
        else:
            nx, ny = grid_cfg["nx"], grid_cfg["ny"]
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
            with torch.no_grad():
                U_pred = model(XY).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star_steady(Xg, Yg).cpu().numpy()
            return save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, "poisson2d_steady")
