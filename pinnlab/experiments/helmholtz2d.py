import math, os, numpy as np
import torch

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class Helmholtz2D(BaseExperiment):
    """
    Time-dependent variant using the wave equation:
        u_tt - c^2 (u_xx + u_yy) = f(x,y,t)

    Choose analytic solution:
        u*(x,y,t) = sin(pi x) sin(pi y) cos(ω t)

    Then:
        u_xx + u_yy = -2 pi^2 u*
        u_tt        = -ω^2 u*
        => residual = (-ω^2 + c^2 * 2 pi^2) u* - f

    If ω = sqrt(2) * pi * c, we can set f ≡ 0 and the solution satisfies the homogeneous equation.
    We enforce Dirichlet BCs from u*, and an initial condition at t = t0.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.device = device
        xa, xb = cfg["domain"]["x"]
        ya, yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]

        self.rect = Rectangle(xa, xb, ya, yb, device)
        self.c = float(cfg.get("c", 1.0))

        omega_cfg = cfg.get("omega", "auto")
        if isinstance(omega_cfg, str) and omega_cfg.lower() == "auto":
            self.omega = math.sqrt(2.0) * math.pi * self.c
        else:
            self.omega = float(omega_cfg)
        
        self.sampling_mode = cfg.get("sampling_mode", "random") # random / grid
        if self.sampling_mode == "grid":
            g = cfg.get("grid", {})
            nx, ny, nt = int(g.get("nx", 100)), int(g.get("ny", 100)), int(g.get("nt", 100))
            
            Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, device)
            XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [nx*ny, 2]
            self._XY = XY
            self._T = torch.linspace(self.t0, self.t1, nt, device=device) # [nt]
            
            # boundary subset (any edge)
            xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
            mask_b = XY[:,0].eq(xa) | XY[:,0].eq(xb) | XY[:,1].eq(ya) | XY[:,1].eq(yb)
            self._XYb = XY[mask_b]  # [nb, 2]

    # ----- analytic fields -----
    def u_star(self, x, y, t):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(self.omega * t)

    def f(self, x, y, t):
        # General forcing for arbitrary omega:
        # f = (-ω^2 + c^2*2π^2) u*, so residual == 0 for the chosen u*.
        coeff = (-self.omega ** 2 + (self.c ** 2) * (2.0 * math.pi ** 2))
        return coeff * self.u_star(x, y, t)

    # ----- sampling -----
    def sample_batch(self, n_f, n_b, n_0):
        if self.sampling_mode == "random":
            X_f_xy = self.rect.sample(n_f)  # [n_f,2]
            t_f = torch.rand(n_f, 1, device=self.device) * (self.t1 - self.t0) + self.t0
            X_f = torch.cat([X_f_xy, t_f], dim=1)

            # Spatial boundary on all 4 edges across random t
            xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
            t_b = torch.rand(4 * n_b, 1, device=self.device) * (self.t1 - self.t0) + self.t0

            y = torch.rand(n_b, 1, device=self.device) * (yb - ya) + ya
            x = torch.rand(n_b, 1, device=self.device) * (xb - xa) + xa

            top    = torch.cat([x, torch.full_like(x, yb)], 1)
            bottom = torch.cat([x, torch.full_like(x, ya)], 1)
            left   = torch.cat([torch.full_like(y, xa), y], 1)
            right  = torch.cat([torch.full_like(y, xb), y], 1)

            X_b_spatial = torch.cat([top, bottom, left, right], dim=0)
            X_b = torch.cat([X_b_spatial, t_b], dim=1)  # [4*n_b, 3]
            u_b = self.u_star(X_b[:, 0:1], X_b[:, 1:2], X_b[:, 2:3])

            # Initial condition at t = t0
            x0y0 = self.rect.sample(n_0)
            t0 = torch.full((n_0, 1), self.t0, device=self.device)
            X_0 = torch.cat([x0y0, t0], dim=1)
            u0 = self.u_star(X_0[:, 0:1], X_0[:, 1:2], X_0[:, 2:3])

            return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}

        elif self.sampling_mode == "grid":
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

    # ----- losses -----
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])       # [N,3] -> (x,y,t)
        u = model(X)

        # first derivatives wrt (x,y,t)
        du = grad_sum(u, X)               # [N,3]
        u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]

        # second derivatives
        d2ux = grad_sum(u_x, X)           # [N,3]
        d2uy = grad_sum(u_y, X)
        d2ut = grad_sum(u_t, X)

        u_xx, u_yy, u_tt = d2ux[:, 0:1], d2uy[:, 1:2], d2ut[:, 2:3]

        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        res = u_tt - (self.c ** 2) * (u_xx + u_yy) - self.f(x, y, t)
        return res.pow(2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub).pow(2)

    def initial_loss(self, model, batch):
        X0, u0 = batch["X_0"], batch["u0"]
        pred = model(X0)
        return (pred - u0).pow(2)

    # ----- eval & plots -----
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        rels = []
        with torch.no_grad():
            for ti in [0, nt // 2, nt - 1]:  # three slices: t0, mid, t1
                tval = ts[ti]
                T = torch.full_like(Xg, tval)
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                U_pred = model(XYT).reshape(nx, ny)
                U_true = self.u_star(Xg, Yg, T)
                rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
                rels.append(rel.item())
        return float(np.mean(rels))

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d

        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        figs = {}
        with torch.no_grad():
            for label, ti in zip(["t0", "tmid", "t1"], [0, nt // 2, nt - 1]):
                tval = ts[ti]
                T = torch.full_like(Xg, tval)
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()

                out = save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"wave2d_{label}")
                figs.update(out)
        return figs
    
    def make_video(self, model, grid, out_dir, fps=10, filename="evolution.mp4"):
        os.makedirs(out_dir, exist_ok=True)
        
