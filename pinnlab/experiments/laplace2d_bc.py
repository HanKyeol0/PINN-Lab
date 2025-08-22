import math, numpy as np, torch
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Rectangle, linspace_2d

class Laplace2D_BC(BaseExperiment):
    """
    Laplace: -Δu = 0 on (x,y) ∈ [xa,xb]×[ya,yb], Dirichlet BC u|∂Ω = g_case(x,y).

    cfg:
      domain: {x:[xa,xb], y:[ya,yb]}
      bc_case: "sine" | "hot_left" | "gaussian" | "checker"
      gaussian: {x0:0.35, y0:0.65, sigma:0.15}   # used if bc_case=="gaussian"
      eval_fd: {nx:129, ny:129, iters:5000, tol:1e-6}  # reference solver
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        xa, xb = cfg["domain"]["x"]
        ya, yb = cfg["domain"]["y"]
        self.rect = Rectangle(xa, xb, ya, yb, device)
        self.case = (cfg.get("bc_case") or "sine").lower() # boundary condition case
        self.gcfg = cfg.get("gaussian", {}) or {} # parameters for gaussian case
        ef = cfg.get("eval_fd", {}) or {}
        self.fd_nx = int(ef.get("nx", 129))
        self.fd_ny = int(ef.get("ny", 129))
        self.fd_iters = int(ef.get("iters", 5000))
        self.fd_tol   = float(ef.get("tol", 1e-6))
        self._ref_cache = {}   # (nx,ny)->numpy array

    # -------- BC library --------
    def g(self, x, y):
        if self.case == "sine":
            return torch.sin(math.pi * x) * torch.sin(math.pi * y)
        if self.case == "hot_left":
            # left edge hot (1), others 0, extended smoothly along boundary:
            return (x*0 + (x == x.min())*1.0).float()  # works on boundary tensors
        if self.case == "gaussian":
            x0 = float(self.gcfg.get("x0", 0.35)); y0 = float(self.gcfg.get("y0", 0.65))
            s  = float(self.gcfg.get("sigma", 0.15))
            r2 = (x - x0)**2 + (y - y0)**2
            return torch.exp(-r2 / (2*s*s))
        if self.case == "checker":
            return torch.sign(torch.sin(3*math.pi*x))*torch.sign(torch.sin(3*math.pi*y))
        raise ValueError(f"Unknown bc_case={self.case}")

    # -------- batching --------
    def sample_batch(self, n_f, n_b, n_0):
        # Interior collocation (x,y)
        X_f = self.rect.sample(n_f)

        # Dirichlet BC samples on 4 edges
        nb = max(1, n_b // 4)
        xa, xb, ya, yb = self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb
        dev = self.rect.device
        y = torch.rand(nb,1,device=dev)*(yb-ya)+ya
        x = torch.rand(nb,1,device=dev)*(xb-xa)+xa
        top    = torch.cat([x, torch.full_like(x, yb)], 1)
        bottom = torch.cat([x, torch.full_like(x, ya)], 1)
        left   = torch.cat([torch.full_like(y, xa), y], 1)
        right  = torch.cat([torch.full_like(y, xb), y], 1)
        X_b = torch.cat([top,bottom,left,right], 0)
        u_b = self.g(X_b[:,0:1], X_b[:,1:2])

        return {"X_f": X_f, "X_b": X_b, "u_b": u_b}

    # -------- losses --------
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])  # (x,y)
        u = model(X)
        du = grad_sum(u, X)          # (u_x,u_y)
        u_x, u_y = du[:,0:1], du[:,1:2]
        d2ux = grad_sum(u_x, X); d2uy = grad_sum(u_y, X)
        u_xx, u_yy = d2ux[:,0:1], d2uy[:,1:2]
        res = -(u_xx + u_yy)         # -Δu
        return res.pow(2)

    def boundary_loss(self, model, batch):
        pred = model(batch["X_b"])
        return (pred - batch["u_b"]).pow(2)

    def initial_loss(self, model, batch):
        return torch.tensor(0.0, device=self.rect.device)  # no time

    # -------- reference (FD) for eval/plots --------
    def _fd_reference(self, nx, ny):
        key = (nx, ny)
        if key in self._ref_cache:
            return self._ref_cache[key]
        xa, xb, ya, yb = map(float, (self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb))
        xs = np.linspace(xa, xb, nx); ys = np.linspace(ya, yb, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        U = np.zeros((nx, ny), dtype=np.float64)

        # set boundary from g
        with torch.no_grad():
            def G(x,y):
                tx = torch.tensor(x, dtype=torch.float32, device=self.rect.device)
                ty = torch.tensor(y, dtype=torch.float32, device=self.rect.device)
                return self.g(tx, ty).detach().cpu().numpy()
            U[0, :]   = G(xa*np.ones_like(ys), ys)
            U[-1, :]  = G(xb*np.ones_like(ys), ys)
            U[:, 0]   = G(xs, ya*np.ones_like(xs))
            U[:, -1]  = G(xs, yb*np.ones_like(xs))

        # Jacobi iterations
        U_new = U.copy()
        for it in range(self.fd_iters):
            U_new[1:-1,1:-1] = 0.25*(U[:-2,1:-1] + U[2:,1:-1] + U[1:-1,:-2] + U[1:-1,2:])
            # keep boundary fixed
            U_new[0,:]=U[0,:]; U_new[-1,:]=U[-1,:]; U_new[:,0]=U[:,0]; U_new[:,-1]=U[:,-1]
            diff = np.max(np.abs(U_new - U))
            U, U_new = U_new, U
            if diff < self.fd_tol: break

        self._ref_cache[key] = U
        return U

    # -------- evaluation & plots --------
    def relative_l2_on_grid(self, model, grid_cfg):
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], 1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx, ny)
        U_true = torch.tensor(self._fd_reference(nx, ny), dtype=U_pred.dtype, device=U_pred.device)
        rel = torch.linalg.norm((U_pred - U_true).reshape(-1)) / torch.linalg.norm(U_true.reshape(-1))
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        from pinnlab.utils.plotting import save_plots_2d
        nx, ny = grid_cfg["nx"], grid_cfg["ny"]
        Xg, Yg = linspace_2d(self.rect.xa, self.rect.xb, self.rect.ya, self.rect.yb, nx, ny, self.rect.device)
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], 1)
        with torch.no_grad():
            U_pred = model(XY).reshape(nx, ny).cpu().numpy()
        U_true = self._fd_reference(nx, ny)
        return save_plots_2d(Xg.cpu().numpy(), Yg.cpu().numpy(), U_true, U_pred, out_dir, f"laplace2d_{self.case}")
