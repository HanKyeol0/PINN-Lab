import math
import torch
import numpy as np
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class AllenCahn2D(BaseExperiment):
    r"""
    Allen–Cahn (2D, time-dependent) with manufactured forcing:

        u_t - ε^2 (u_xx + u_yy) + (u^3 - u) = f(x,y,t),  (x,y)∈[xa,xb]×[ya,yb], t∈[t0,t1]

    Manufactured solution:
        u*(x,y,t) = sin(π x) sin(π y) cos(ω t)

    Then
        f = u*_t - ε^2 Δu* + (u*^3 - u*)
    with Dirichlet BC/IC from u*.

    cfg:
      domain: {x: [xa, xb], y: [ya, yb], t: [t0, t1]}
      eps: 0.01
      omega: 2.0
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]; ya, yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]
        self.rect  = Rectangle(xa, xb, ya, yb, device)
        self.eps   = float(cfg.get("eps", 0.01))
        self.omega = float(cfg.get("omega", 2.0))

    # ----- manufactured truth -----
    def u_star(self, x, y, t):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(self.omega * t)

    def f(self, x, y, t):
        u   = self.u_star(x, y, t)
        ut  = -self.omega * torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.sin(self.omega * t)
        lap = -2.0 * (math.pi**2) * self.u_star(x, y, t)  # Δu* = u_xx + u_yy
        return ut - (self.eps**2) * lap + (u**3 - u)

    # ----- batching -----
    def sample_batch(self, n_f: int, n_b: int, n_0: int):
        device = self.rect.device

        # interior (x,y,t)
        XY  = self.rect.sample(n_f)
        t_f = torch.rand(n_f, 1, device=device) * (self.t1 - self.t0) + self.t0
        X_f = torch.cat([XY, t_f], dim=1)

        # boundary on 4 edges (Dirichlet from u*)
        nb = max(1, n_b // 4)
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
        t_b = lambda m: torch.rand(m,1,device=device)*(self.t1-self.t0)+self.t0

        # x=xa|xb
        yL = torch.rand(nb,1,device=device)*(yb-ya)+ya; TL = t_b(nb)
        yR = torch.rand(nb,1,device=device)*(yb-ya)+ya; TR = t_b(nb)
        Xb_L = torch.cat([torch.full_like(yL, xa), yL, TL], dim=1)
        Xb_R = torch.cat([torch.full_like(yR, xb), yR, TR], dim=1)

        # y=ya|yb
        xB = torch.rand(nb,1,device=device)*(xb-xa)+xa; TB = t_b(nb)
        xT = torch.rand(nb,1,device=device)*(xb-xa)+xa; TT = t_b(nb)
        Xb_B = torch.cat([xB, torch.full_like(xB, ya), TB], dim=1)
        Xb_T = torch.cat([xT, torch.full_like(xT, yb), TT], dim=1)

        X_b = torch.cat([Xb_L, Xb_R, Xb_B, Xb_T], dim=0)
        u_b = self.u_star(X_b[:,0:1], X_b[:,1:2], X_b[:,2:3])

        # initial condition at t = t0
        XY0 = self.rect.sample(n_0)
        X_0 = torch.cat([XY0, torch.full((n_0,1), self.t0, device=device)], dim=1)
        u0  = self.u_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])

        return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}

    # ----- losses -----
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])                # (x,y,t)
        x, y, t = X[:,0:1], X[:,1:2], X[:,2:3]
        u   = model(X)
        du  = grad_sum(u, X)                       # (u_x, u_y, u_t)
        u_x, u_y, u_t = du[:,0:1], du[:,1:2], du[:,2:3]
        d2u_x = grad_sum(u_x, X)                   # (u_xx, u_xy, u_xt)
        d2u_y = grad_sum(u_y, X)                   # (u_yx, u_yy, u_yt)
        u_xx, u_yy = d2u_x[:,0:1], d2u_y[:,1:2]
        res = u_t - (self.eps**2) * (u_xx + u_yy) + (u**3 - u) - self.f(x, y, t)
        return res.pow(2)

    def boundary_loss(self, model, batch):
        return (model(batch["X_b"]) - batch["u_b"]).pow(2)

    def initial_loss(self, model, batch):
        return (model(batch["X_0"]) - batch["u0"]).pow(2)

    # ----- eval / plots -----
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
        idxs = [0, nt//2, nt-1] if nt >= 3 else list(range(nt))
        rels = []
        with torch.no_grad():
            for ti in idxs:
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], 1)
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
            for label, ti in zip(["t0","tmid","t1"], [0, nt//2, nt-1]):
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], 1)
                U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()
                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"allencahn2d_{label}"))
        return figs
