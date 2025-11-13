import math, os
import numpy as np
from matplotlib import pyplot as plt
import imageio
import torch

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class NavierStokes2D(BaseExperiment):
    """
    2D incompressible Navier–Stokes (x,y,t), Taylor–Green vortex (periodic in x,y).

    Momentum:
        u_t + u u_x + v u_y + (1/ρ) p_x - ν (u_xx + u_yy) = 0
        v_t + u v_x + v v_y + (1/ρ) p_y - ν (v_xx + v_yy) = 0
    Continuity:
        u_x + v_y = 0

    Analytic (TGV) solution:
        u*(x,y,t) =  U0 cos(kx) sin(ky) exp(-2 ν k^2 t)
        v*(x,y,t) = -U0 sin(kx) cos(ky) exp(-2 ν k^2 t)
        p*(x,y,t) =  ρ U0^2 / 4 [cos(2kx) + cos(2ky)] exp(-4 ν k^2 t)

    We enforce:
      - periodic BCs (paired equality at x=xa/xb and y=ya/yb)
      - IC at t = t0 from the analytic solution
      - PDE residual (momentum + continuity) at interior collocation points
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        ya, yb = cfg["domain"]["y"]
        self.t0, self.t1 = cfg["domain"]["t"]

        self.rect = Rectangle(xa, xb, ya, yb, device)
        self.nu   = float(cfg.get("nu", 0.01))
        self.rho  = float(cfg.get("rho", 1.0))
        self.U0   = float(cfg.get("U0", 1.0))
        self.k    = float(cfg.get("k", 1.0))

        w = cfg.get("weights", {}) or {}
        self.w_mom = float(w.get("momentum", 1.0))
        self.w_div = float(w.get("continuity", 1.0))

        self.input_dim = 3  # (x,y,t)

    # -------- Analytic fields (Taylor–Green) --------
    def u_star(self, x, y, t):
        return self.U0 * torch.cos(self.k * x) * torch.sin(self.k * y) * torch.exp(-2.0 * self.nu * (self.k**2) * t)

    def v_star(self, x, y, t):
        return -self.U0 * torch.sin(self.k * x) * torch.cos(self.k * y) * torch.exp(-2.0 * self.nu * (self.k**2) * t)

    def p_star(self, x, y, t):
        amp = 0.25 * self.rho * (self.U0 ** 2)
        return amp * (torch.cos(2.0 * self.k * x) + torch.cos(2.0 * self.k * y)) * torch.exp(-4.0 * self.nu * (self.k**2) * t)

    # -------- Sampling --------
    def sample_batch(self, n_f, n_b, n_0):
        device = self.rect.device
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb

        # Interior collocation: (x,y,t)
        XY = self.rect.sample(n_f)
        t_f = torch.rand(n_f, 1, device=device) * (self.t1 - self.t0) + self.t0
        X_f = torch.cat([XY, t_f], dim=1)

        # Periodic BC pairs (x: left/right, y: bottom/top) across random t
        nb = max(1, n_b // 2)
        t_b1 = torch.rand(nb, 1, device=device) * (self.t1 - self.t0) + self.t0
        y1   = torch.rand(nb, 1, device=device) * (yb - ya) + ya
        x_left  = torch.full((nb,1), xa, device=device);  x_right = torch.full((nb,1), xb, device=device)
        X_lr_L = torch.cat([x_left,  y1, t_b1], dim=1)   # (xa, y, t)
        X_lr_R = torch.cat([x_right, y1, t_b1], dim=1)   # (xb, y, t)

        t_b2 = torch.rand(nb, 1, device=device) * (self.t1 - self.t0) + self.t0
        x2   = torch.rand(nb, 1, device=device) * (xb - xa) + xa
        y_bot = torch.full((nb,1), ya, device=device);  y_top = torch.full((nb,1), yb, device=device)
        X_bt_B = torch.cat([x2, y_bot, t_b2], dim=1)    # (x, ya, t)
        X_bt_T = torch.cat([x2, y_top, t_b2], dim=1)    # (x, yb, t)

        # Merge pairs so boundary_loss can do a single equality loss
        X_bp1 = torch.cat([X_lr_L, X_bt_B], dim=0)
        X_bp2 = torch.cat([X_lr_R, X_bt_T], dim=0)

        # Initial condition at t = t0
        XY0 = self.rect.sample(n_0)
        t0  = torch.full((n_0, 1), self.t0, device=device)
        X_0 = torch.cat([XY0, t0], dim=1)
        u0  = self.u_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])
        v0  = self.v_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])
        p0  = self.p_star(X_0[:,0:1], X_0[:,1:2], X_0[:,2:3])
        Y0  = torch.cat([u0, v0, p0], dim=1)

        return {"X_f": X_f, "X_bp1": X_bp1, "X_bp2": X_bp2, "X_0": X_0, "Y0": Y0}

    # -------- Losses --------
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])        # [N,3] (x,y,t)
        out = model(X)                     # [N,3] -> (u,v,p)
        u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]

        du = grad_sum(u, X)                # [N,3] -> (u_x,u_y,u_t)
        dv = grad_sum(v, X)
        dp = grad_sum(p, X)

        u_x, u_y, u_t = du[:,0:1], du[:,1:2], du[:,2:3]
        v_x, v_y, v_t = dv[:,0:1], dv[:,1:2], dv[:,2:3]
        p_x, p_y, _   = dp[:,0:1], dp[:,1:2], dp[:,2:3]

        d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        d2vx = grad_sum(v_x, X); d2vy = grad_sum(v_y, X)
        u_xx, u_yy = d2ux[:,0:1], d2uy[:,1:2]
        v_xx, v_yy = d2vx[:,0:1], d2vy[:,1:2]

        # Momentum residuals (bring all to left-hand side)
        res_u = u_t + u * u_x + v * u_y + (1.0/self.rho) * p_x - self.nu * (u_xx + u_yy)
        res_v = v_t + u * v_x + v * v_y + (1.0/self.rho) * p_y - self.nu * (v_xx + v_yy)

        # Continuity residual
        div = u_x + v_y

        # Weighted squared residuals
        return self.w_mom * (res_u**2 + res_v**2) + self.w_div * (div**2)

    def boundary_loss(self, model, batch):
        # Periodic equality: model(X_bp1) == model(X_bp2) for all outputs (u,v,p)
        if "X_bp1" not in batch: 
            return torch.tensor(0.0, device=self.device)
        A = model(batch["X_bp1"])
        B = model(batch["X_bp2"])
        return (A - B).pow(2)

    def initial_loss(self, model, batch):
        if "X_0" not in batch:
            return torch.tensor(0.0, device=self.device)
        pred = model(batch["X_0"])     # [N,3]
        return (pred - batch["Y0"]).pow(2)

    # -------- Evaluation --------
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)
        idxs = [0, nt//2, nt-1] if nt >= 3 else list(range(nt))
        rels = []
        with torch.no_grad():
            for ti in idxs:
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                out = model(XYT).reshape(nx, ny, 3)
                U_pred = out[:,:,0]; V_pred = out[:,:,1]
                U_true = self.u_star(Xg, Yg, T); V_true = self.v_star(Xg, Yg, T)
                rel_u = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
                rel_v = torch.linalg.norm((V_pred - V_true).reshape(-1)) / torch.linalg.norm(V_true.reshape(-1))
                rels.append(0.5*(rel_u + rel_v).item())
        return float(np.mean(rels))

    # -------- Plots --------
    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        nx, ny, nt = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        figs = {}
        with torch.no_grad():
            for label, ti in zip(["t0","tmid","t1"], [0, nt//2, nt-1]):
                T = torch.full_like(Xg, ts[ti])
                XYT = torch.stack([Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1)
                out = model(XYT).reshape(nx, ny, 3).cpu().numpy()

                U_pred = out[:,:,0]; V_pred = out[:,:,1]; P_pred = out[:,:,2]
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()
                V_true = self.v_star(Xg, Yg, T).cpu().numpy()
                P_true = self.p_star(Xg, Yg, T).cpu().numpy()

                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"ns2d_u_{label}"))
                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), V_true, V_pred, out_dir, f"ns2d_v_{label}"))
                figs.update(save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), P_true, P_pred, out_dir, f"ns2d_p_{label}"))
        return figs

    def make_video(self, model, grid, out_dir, fps=10, filename="evolution.mp4"):
        """
        Make a video showing the temporal evolution of (u, v, p) on the 2D domain.

        - grid: dict with keys {"nx", "ny", "nt"}
        - Produces a video with 2x3 panels:
            top row:  u_pred, v_pred, p_pred
            bottom:   u_true, v_true, p_true
        """
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)

        nx = int(grid.get("nx", 64))
        ny = int(grid.get("ny", 64))
        nt = int(grid.get("nt", 50))

        device = self.rect.device
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb,
                             self.rect.ya, self.rect.yb,
                             nx, ny, device)
        ts = torch.linspace(self.t0, self.t1, nt, device=device)

        # Choose writer depending on extension
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".gif":
            writer = imageio.get_writer(path, mode="I", duration=1.0 / fps)
        else:
            # mp4 (default) – codec arg may require ffmpeg/libx264 in the system
            writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)

        with writer:
            for ti, tval in enumerate(ts):
                T = torch.full_like(Xg, tval)

                # Build (x,y,t) query points
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )

                with torch.no_grad():
                    out = model(XYT).reshape(nx, ny, 3).cpu().numpy()
                    U_pred = out[:, :, 0]
                    V_pred = out[:, :, 1]
                    P_pred = out[:, :, 2]

                    U_true = self.u_star(Xg, Yg, T).cpu().numpy()
                    V_true = self.v_star(Xg, Yg, T).cpu().numpy()
                    P_true = self.p_star(Xg, Yg, T).cpu().numpy()

                # 2x3 panel: (u,v,p) for pred (top) and true (bottom)
                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                fields_pred = [U_pred, V_pred, P_pred]
                fields_true = [U_true, V_true, P_true]
                titles = ["u", "v", "p"]

                for col in range(3):
                    # predicted
                    ax = axes[0, col]
                    im = ax.imshow(
                        fields_pred[col],
                        origin="lower",
                        extent=[self.rect.xa, self.rect.xb,
                                self.rect.ya, self.rect.yb],
                    )
                    ax.set_title(f"pred {titles[col]}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    fig.colorbar(im, ax=ax, shrink=0.7)

                    # true
                    ax = axes[1, col]
                    im = ax.imshow(
                        fields_true[col],
                        origin="lower",
                        extent=[self.rect.xa, self.rect.xb,
                                self.rect.ya, self.rect.yb],
                    )
                    ax.set_title(f"true {titles[col]}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    fig.colorbar(im, ax=ax, shrink=0.7)

                fig.suptitle(f"Navier–Stokes 2D  t = {float(tval):.3f}", y=0.98)
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                # Convert figure to RGB array for imageio
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                writer.append_data(frame)

        return path