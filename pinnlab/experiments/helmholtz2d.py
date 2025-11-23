import math, os, numpy as np, io
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d
from pinnlab.data.noise import get_noise
from pinnlab.utils.ebm import EBM

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

        noise_cfg = cfg.get("noise", None)
        self.use_data = bool(noise_cfg.get("enabled", False))
        self.noise_cfg = noise_cfg
        self.n_data_total = int(noise_cfg.get("n_train", 0))
        self.n_data_batch = int(
            noise_cfg.get("batch_size", 1000)
        )
        
        self.X_data = None
        self.y_data = None
        self.y_clean = None
        self.noise_model = None
        
        if self.use_data and self.n_data_total > 0:
            print(f"Initializing noisy data with {self.n_data_total} points.")
            self._init_noisy_dataset()

        ebm_cfg = cfg.get("ebm", {}) or {}
        self.use_ebm = bool(ebm_cfg.get("enabled", False))
        self.use_nll = bool(ebm_cfg.get("use_nll", False))
        if self.use_ebm:
            self.ebm = EBM(
                hidden_dim=ebm_cfg.get("hidden_dim", 32),
                depth=ebm_cfg.get("depth", 3),
                num_grid=ebm_cfg.get("num_grid", 256),
                max_range_factor=ebm_cfg.get("max_range_factor", 2.5),
                lr=ebm_cfg.get("lr", 1e-3),
                device=device,
            )
        else:
            self.ebm = None
        
        self.use_data_loss_balancer = bool(cfg.get("data_loss_balancer", {}).get("use_loss_balancer", False))

        # ----- Optional trainable offset θ0 for non-zero mean noise (PINN-off style) -----
        offset_cfg = cfg.get("offset", {}) or {}
        self.use_offset = bool(offset_cfg.get("enabled", False))
        if self.use_offset:
            init = float(offset_cfg.get("init", 0.0))
            # scalar parameter θ0 that will only be used in the DATA term
            self.offset = torch.nn.Parameter(
                torch.tensor(init, dtype=torch.float32, device=device)
            )
            print(f"[Helmholtz2D] Using trainable data offset θ0, init={init}")
        else:
            self.offset = None

    # ----- analytic fields -----
    def u_star(self, x, y, t):
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(self.omega * t)

    def f(self, x, y, t):
        # General forcing for arbitrary omega:
        # f = (-ω^2 + c^2*2π^2) u*, so residual == 0 for the chosen u*.
        coeff = (-self.omega ** 2 + (self.c ** 2) * (2.0 * math.pi ** 2))
        return coeff * self.u_star(x, y, t)
    
    def _init_noisy_dataset(self):
        """
        Generate a fixed set of noisy measurements:
            X_data ~ Uniform(domain x time)
            y_clean = u_star(X_data)
            y_data  = y_clean + epsilon,     epsilon ~ noise distribution (from PINN-EBM)
        """
        kind = self.noise_cfg.get("kind", "3G")     # 'G', 'u', '3G', ...
        n = self.n_data_total

        # Sample input locations
        XY = self.rect.sample(n)  # [n, 2] in (x, y)
        t = torch.rand(n, 1, device=self.device) * (self.t1 - self.t0) + self.t0
        X = torch.cat([XY, t], dim=1)  # [n, 3]

        with torch.no_grad():
            u_clean = self.u_star(X[:, 0:1], X[:, 1:2], X[:, 2:3])  # [n, 1]

        # Determine noise scale "f" as in PINN-EBM: a factor times average magnitude of u*
        base_scale = float(self.noise_cfg.get("scale", 0.1))  # relative to mean |u|
        mean_level = float(u_clean.abs().mean().detach().cpu())
        f = base_scale * (mean_level if mean_level > 0 else 1.0)

        # Build noise distribution; this uses the PINN-EBM function
        self.noise_model = get_noise(kind, f, pars=0)

        # In PINN-EBM, sample() often expects a shape list (e.g. [N])
        eps = self.noise_model.sample(n).to(self.device).view(-1, 1)  # [n, 1]

        y_noisy = u_clean + eps

        self.X_data = X
        self.y_clean = u_clean
        self.y_data = y_noisy

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

            batch = {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}

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

            batch = {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}

        # --- NEW: attach noisy data mini-batch for data loss ---
        if self.use_data and self.X_data is not None and self.n_data_batch > 0:
            n = self.X_data.size(0)
            k = min(self.n_data_batch, n)
            idx = torch.randint(0, n, (k,), device=self.device)
            batch["X_d"] = self.X_data[idx]
            batch["y_d"] = self.y_data[idx]

        return batch

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
    
    def data_loss(self, model, batch, phase=1):
        if "X_d" not in batch or "y_d" not in batch:
            return torch.tensor(0.0, device=self.device)

        X_d = batch["X_d"]
        y_d = batch["y_d"]

        # Raw PINN prediction (this is what PDE/BC/IC see)
        u_raw = model(X_d)  # [N, 1]

        # For the DATA term, optionally add scalar offset θ0
        if getattr(self, "use_offset", False) and self.offset is not None:
            u_data = u_raw + self.offset  # broadcast θ0
        else:
            u_data = u_raw

        # Residuals for data and for EBM
        err = u_data - y_d              # [N, 1] (model - noisy data)
        residual = (y_d - u_data)       # [N, 1] (data - model), used as "noise"
        
        if phase == 0:
            if self.ebm is not None and residual.numel() > 0:
                # Detach so EBM training does not backprop through PINN/θ0
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual.detach())
                batch["ebm_nll"] = nll_ebm_mean

                if self.use_data_loss_balancer:
                    print("[data_loss] Using EBM-based data loss balancer.")
                    # Per-point reliability weights from EBM (mean ≈ 1)
                    w = self.ebm.pointwise_weights(residual.detach())  # [N, 1]
                    
                    loss_per_sample = (w * err.pow(2)).view(-1)
                else:
                    print("[data_loss] NOT using EBM-based data loss balancer.")
                    loss_per_sample = err.pow(2).view(-1)
                return loss_per_sample
            
        elif phase == 1:
            return err.pow(2).view(-1)
        
        elif phase == 2:
            if self.ebm is not None and residual.numel() > 0:
                # Detach so EBM training does not backprop through PINN/θ0
                nll_ebm, nll_ebm_mean = self.ebm.train_step(residual.detach())
                batch["ebm_nll"] = nll_ebm_mean

                if self.use_nll:
                    print("[data_loss] Using EBM-based NLL data loss.")
                    data_loss = nll_ebm
                else:
                    print("[data_loss] Using MSE data loss.")
                    data_loss = err.pow(2)
                if self.use_data_loss_balancer:
                    print("[data_loss] Using EBM-based data loss balancer.")
                    # Per-point reliability weights from EBM (mean ≈ 1)
                    w = self.ebm.pointwise_weights(residual.detach())  # [N, 1]

                    loss_per_sample = (w * data_loss).view(-1)
                else:
                    print("[data_loss] NOT using EBM-based data loss balancer.")
                    loss_per_sample = data_loss.view(-1)

                return loss_per_sample
        
        else:
            raise ValueError(f"Invalid phase for data_loss: {phase}")
    
    def extra_params(self):
        """Experiment-specific trainable parameters (e.g., θ0)."""
        if isinstance(getattr(self, "offset", None), torch.nn.Parameter):
            return [self.offset]
        return []

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
    
    def make_video(self, model, grid, out_dir, fps=10, filename="final_evolution.mp4"):
        """
        Make a video over t ∈ [t0, t1] with 4 panels:

            [0,0] Noisy data over entire domain: u*(x,y,t) + ε(x,y,t)
            [0,1] True solution u*(x,y,t)
            [1,0] Predicted solution u_hat(x,y,t)
            [1,1] Absolute error |u* - u_hat|

        - Color scales for u / noisy panels are consistent across time.
        - Error color scale is also global across time.
        - If noise is disabled (no self.noise_model), the noisy panel = true solution.

        Args:
            model: trained PINN model
            grid: dict with keys {"nx","ny","nt"}
            out_dir: directory to save video
            fps: frames per second
            filename: video file name (default: final_evolution.mp4)

        Returns:
            Full path to the main evolution video.
        """
        os.makedirs(out_dir, exist_ok=True)
        nx = int(grid.get("nx", 100))
        ny = int(grid.get("ny", 100))
        nt = int(grid.get("nt", 100))

        # Build spatial grid
        Xg, Yg = linspace_2d(
            self.rect.xa, self.rect.xb,
            self.rect.ya, self.rect.yb,
            nx, ny, self.rect.device
        )
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        extent = [float(self.rect.xa), float(self.rect.xb),
                float(self.rect.ya), float(self.rect.yb)]

        # Check if we have a noise model
        has_noise_model = getattr(self, "noise_model", None) is not None

        # ---------- First pass: determine global color ranges ----------
        vmin, vmax = None, None
        err_max = 0.0

        model.eval()
        with torch.no_grad():
            for tval in ts:
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                U_pred = model(XYT).reshape(nx, ny)    # [nx, ny]
                U_true = self.u_star(Xg, Yg, T)        # [nx, ny]

                if has_noise_model:
                    eps = self.noise_model.sample(nx * ny).to(self.rect.device)
                    eps = eps.view(nx, ny)
                    U_noisy = U_true + eps
                else:
                    U_noisy = U_true

                # Global color range for all "solution-like" panels
                umin = min(U_true.min().item(), U_pred.min().item(), U_noisy.min().item())
                umax = max(U_true.max().item(), U_pred.max().item(), U_noisy.max().item())
                vmin = umin if vmin is None else min(vmin, umin)
                vmax = umax if vmax is None else max(vmax, umax)

                # Error scale
                err_max = max(err_max, (U_true - U_pred).abs().max().item())

        # Safety
        if err_max <= 0:
            err_max = 1e-6

        # ---------- Second pass: render frames ----------
        frames = []
        with torch.no_grad():
            for tval in ts:
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )

                U_pred = model(XYT).reshape(nx, ny).cpu().numpy()
                U_true = self.u_star(Xg, Yg, T).cpu().numpy()

                if has_noise_model:
                    eps = self.noise_model.sample(nx * ny).to(self.rect.device)
                    eps = eps.view(nx, ny).cpu().numpy()
                    U_noisy = U_true + eps
                else:
                    U_noisy = U_true  # fallback: no noise → same as true

                U_err = np.abs(U_true - U_pred)

                fig = plt.figure(figsize=(12, 12), dpi=120)
                plt.suptitle(f"t = {float(tval):.5f}")

                # [0,0] Noisy data over whole domain
                ax1 = plt.subplot(2, 2, 1)
                im1 = ax1.imshow(
                    U_noisy.T, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, aspect="auto"
                )
                ax1.set_title("Noisy data: u*(x,y,t) + ε")
                ax1.set_xlabel("x"); ax1.set_ylabel("y")
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # [0,1] True solution
                ax2 = plt.subplot(2, 2, 2)
                im2 = ax2.imshow(
                    U_true.T, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, aspect="auto"
                )
                ax2.set_title("True solution u*(x,y,t)")
                ax2.set_xlabel("x"); ax2.set_ylabel("y")
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                # [1,0] Predicted solution
                ax3 = plt.subplot(2, 2, 3)
                im3 = ax3.imshow(
                    U_pred.T, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, aspect="auto"
                )
                ax3.set_title("Predicted solution û(x,y,t)")
                ax3.set_xlabel("x"); ax3.set_ylabel("y")
                fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

                # [1,1] Absolute error
                ax4 = plt.subplot(2, 2, 4)
                im4 = ax4.imshow(
                    U_err.T, origin="lower", extent=extent,
                    vmin=0.0, vmax=err_max, aspect="auto"
                )
                ax4.set_title("|Error| = |u* - û|")
                ax4.set_xlabel("x"); ax4.set_ylabel("y")
                fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Convert figure to frame
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape(h, w, 3)
                frames.append(frame.copy())
                plt.close(fig)

        # ---------- Write video ----------
        ext = os.path.splitext(filename)[1].lower()
        path = os.path.join(out_dir, filename)
        if ext == ".gif":
            imageio.mimsave(path, frames, fps=fps)
        elif ext == ".mp4":
            writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
            for fr in frames:
                writer.append_data(fr)
            writer.close()
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        # (optional) still call your noise-distribution video helper if you keep it
        try:
            self._make_noise_videos(model, grid, out_dir, fps, filename)
        except Exception as e:
            print(f"[make_video] Warning: noise video failed with error: {e}")

        return path
    
    def _make_noise_videos(self, model, grid, out_dir, fps, filename):
        """
        Create a single video visualizing noise over the whole domain.

        For each time slice t_k, the figure has 4 panels:

            [0,0]  True noise field ε*(x,y,t_k) sampled on the full grid
            [0,1]  Residual field r(x,y,t_k) = y_noisy - u_pred

                   where y_noisy = u*(x,y,t_k) + ε*(x,y,t_k)

            [1,0]  Histograms of ε* and r on the same axes
            [1,1]  True noise pdf vs EBM pdf on the same axes

        This satisfies:
          - True vs predicted noise distributions in the SAME figure.
          - Noise distribution measured over the WHOLE domain, not just sampled data points.
        """
        # Need both true noise model and EBM to make this meaningful
        if getattr(self, "noise_model", None) is None:
            return None, None
        if getattr(self, "ebm", None) is None:
            # You could still visualize true noise only, but here we require both.
            return None, None

        base, ext = os.path.splitext(filename)
        if ext == "":
            ext = ".mp4"

        os.makedirs(out_dir, exist_ok=True)

        nx = int(grid.get("nx", 50))
        ny = int(grid.get("ny", 50))
        nt = int(grid.get("nt", 50))

        # Spatial grid
        Xg, Yg = linspace_2d(
            self.rect.xa, self.rect.xb,
            self.rect.ya, self.rect.yb,
            nx, ny, self.rect.device
        )
        ts = torch.linspace(self.t0, self.t1, nt, device=self.rect.device)

        extent = [float(self.rect.xa), float(self.rect.xb),
                  float(self.rect.ya), float(self.rect.yb)]

        frames = []

        model.eval()
        with torch.no_grad():
            for tval in ts:
                # Build full space-time grid
                T = torch.full_like(Xg, tval)
                XYT = torch.stack(
                    [Xg.reshape(-1), Yg.reshape(-1), T.reshape(-1)], dim=1
                )  # [nx*ny, 3]

                # True solution and prediction on full grid
                U_true = self.u_star(Xg, Yg, T)                  # [nx, ny]
                U_pred = model(XYT).reshape(nx, ny)              # [nx, ny]

                # Sample true noise on full grid (whole domain)
                eps_true_flat = self.noise_model.sample(nx * ny).to(self.rect.device)
                eps_true = eps_true_flat.view(nx, ny)            # [nx, ny]

                # Noisy observations and residuals
                Y_noisy = U_true + eps_true                      # [nx, ny]
                R_field = (Y_noisy - U_pred)                     # [nx, ny]

                # Flatten for 1D distributions
                eps_flat = eps_true.detach().cpu().numpy().reshape(-1)
                res_flat = R_field.detach().cpu().numpy().reshape(-1)

                # Range for plots / pdf grid
                max_val = max(
                    float(np.max(np.abs(eps_flat))) if eps_flat.size > 0 else 0.0,
                    float(np.max(np.abs(res_flat))) if res_flat.size > 0 else 0.0,
                    1e-3,
                )
                R = 1.5 * max_val
                r_grid = np.linspace(-R, R, 200, dtype=np.float32)
                r_torch = torch.from_numpy(r_grid).float().to(self.rect.device)

                # True noise pdf
                pdf_true = None
                if hasattr(self.noise_model, "pdf"):
                    r_cpu = torch.from_numpy(r_grid).float()  # CPU tensor
                    pdf_true_tensor = self.noise_model.pdf(r_cpu)  # all CPU ops inside noise.py
                    if isinstance(pdf_true_tensor, torch.Tensor):
                        pdf_true = pdf_true_tensor.detach().cpu().numpy()
                    else:
                        pdf_true = np.asarray(pdf_true_tensor)
                        
                # EBM pdf (from log q_theta)
                with torch.no_grad():
                    log_q = self.ebm(r_torch.unsqueeze(-1)).squeeze(-1)  # [200]
                    log_q = log_q - log_q.max()  # shift for numerical stability
                    pdf_unn = torch.exp(log_q)   # unnormalized
                    Z = torch.trapezoid(pdf_unn, r_torch)
                    pdf_ebm = (pdf_unn / (Z + 1e-12)).cpu().numpy()

                # ---- Build figure ----
                fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=120)
                fig.suptitle(f"Noise distributions at t = {float(tval):.4f}")

                # [0,0] True noise field ε*(x,y,t)
                im0 = axes[0, 0].imshow(
                    eps_true.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 0].set_title("True noise field ε*(x,y,t)")
                axes[0, 0].set_xlabel("x")
                axes[0, 0].set_ylabel("y")
                fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

                # [0,1] Residual field r(x,y,t)
                im1 = axes[0, 1].imshow(
                    R_field.cpu().numpy().T,
                    origin="lower",
                    extent=extent,
                    vmin=-R,
                    vmax=R,
                    aspect="auto",
                    cmap="coolwarm",
                )
                axes[0, 1].set_title("Residual field r(x,y,t) = y_noisy - u_pred")
                axes[0, 1].set_xlabel("x")
                axes[0, 1].set_ylabel("y")
                fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

                # [1,0] Histograms of ε* and r on same axes
                axes[1, 0].hist(
                    eps_flat, bins=40, density=True,
                    alpha=0.5, label="true noise sample ε*",
                )
                axes[1, 0].hist(
                    res_flat, bins=40, density=True,
                    alpha=0.5, label="residual sample r",
                )
                axes[1, 0].set_xlim(-R, R)
                axes[1, 0].set_xlabel("value")
                axes[1, 0].set_ylabel("density")
                axes[1, 0].set_title("Empirical distributions over whole domain")
                axes[1, 0].legend(loc="upper right", fontsize=8)

                # [1,1] True pdf vs EBM pdf on same axes
                if pdf_true is not None:
                    axes[1, 1].plot(
                        r_grid, pdf_true,
                        label="true noise pdf",
                        linewidth=1.5,
                    )
                axes[1, 1].plot(
                    r_grid, pdf_ebm,
                    label="EBM pdf",
                    linestyle="--",
                    linewidth=1.5,
                )
                axes[1, 1].set_xlim(-R, R)
                axes[1, 1].set_xlabel("value")
                axes[1, 1].set_ylabel("density")
                axes[1, 1].set_title("True vs EBM noise pdf")
                axes[1, 1].legend(loc="upper right", fontsize=8)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Convert figure to frame
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape(h, w, 3)
                frames.append(frame.copy())
                plt.close(fig)

        if not frames:
            return None, None

        # Save video
        path_noise = os.path.join(out_dir, f"{base}_noise_dist{ext}")
        if ext == ".gif":
            imageio.mimsave(path_noise, frames, fps=fps)
        elif ext == ".mp4":
            writer = imageio.get_writer(path_noise, fps=fps, codec="libx264", quality=7)
            for fr in frames:
                writer.append_data(fr)
            writer.close()
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        # For backward-compatibility, return (path_true, path_ebm) style if you want:
        return path_noise, None
