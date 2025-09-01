import os, time, yaml, argparse, sys
import torch
import wandb
from tqdm import trange
from pinnlab.registry import get_model, get_experiment
from pinnlab.utils.seed import seed_everything
from pinnlab.utils.early_stopping import EarlyStopping
from pinnlab.utils.plotting import save_plots_1d, save_plots_2d
from pinnlab.utils.wandb_utils import setup_wandb, wandb_log, wandb_finish
from pinnlab.utils.gradflow import GradientFlowLogger

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)

def main(args):
    base_cfg = load_yaml(args.common_config)
    model_cfg = load_yaml(args.model_config)
    exp_cfg   = load_yaml(args.exp_config)

    # Allow experiment to override in/out dims if needed
    in_features  = exp_cfg.get("in_features", model_cfg.get("in_features"))
    out_features = exp_cfg.get("out_features", model_cfg.get("out_features"))
    model_cfg["in_features"]  = in_features
    model_cfg["out_features"] = out_features

    seed_everything(base_cfg["seed"])

    device = torch.device(base_cfg["device"] if torch.cuda.is_available() else "cpu")
    exp = get_experiment(args.experiment_name)(exp_cfg, device)
    model = get_model(args.model_name)(model_cfg).to(device)

    # Optimizer
    opt_cfg = base_cfg["train"]["optimizer"]
    if opt_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])
    else:
        raise ValueError("Only Adam is wired in, add more in train.py.")

    # Logging dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_cfg["log"]["out_dir"], f"{args.experiment_name}_{args.model_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    _save_yaml(os.path.join(out_dir, "config.yaml"), {
        "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
    })

    # WandB
    if base_cfg["log"]["wandb"]["enabled"]:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(project = base_cfg["log"]["wandb"]["project"],
                   name = f"{args.experiment_name}_{args.model_name}_{ts}")
        run = setup_wandb(base_cfg["log"]["wandb"], args, out_dir, config={
            "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
        })

    # gradient flow logger
    gf = GradientFlowLogger(
        model, out_dir,
        enable=base_cfg.get("gradflow", {}).get("enabled", True),
        every=int(base_cfg.get("gradflow", {}).get("every", 1000)),
        store_vectors=bool(base_cfg.get("gradflow", {}).get("store_vectors", False)),
        max_keep_per_layer=int(base_cfg.get("gradflow", {}).get("max_keep", 20)),
        wandb_hist=bool(base_cfg.get("gradflow", {}).get("wandb_hist", False)),
        plot_cfg=base_cfg.get("gradflow", {}).get("plot", {})
    )

    epochs = base_cfg["train"]["epochs"]
    eval_every = int(base_cfg.get("eval").get("every", 100))

    # Early stopping
    es_cfg = base_cfg["train"]["early_stopping"]
    early = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"], eval_every=eval_every) if es_cfg["enabled"] else None
    best_state = None
    best_metric = float("inf")

    # Training loop
    w_f = base_cfg["train"]["loss_weights"]["f"]
    w_b = base_cfg["train"]["loss_weights"]["b"]
    w_ic= base_cfg["train"]["loss_weights"]["ic"]

    n_f = exp_cfg.get("batch", {}).get("n_f", base_cfg["train"]["batch"]["n_f"])
    n_b = exp_cfg.get("batch", {}).get("n_b", base_cfg["train"]["batch"]["n_b"])
    n_0 = exp_cfg.get("batch", {}).get("n_0", base_cfg["train"]["batch"]["n_0"])

    use_tty = sys.stdout.isatty()
    pbar = trange(
        epochs,
        desc="Training",
        ncols=120,
        dynamic_ncols=True,
        leave=False,          # don't leave old bars behind
        disable=not use_tty,  # if output is piped, avoid multiline spam
    )
    gf_stop = base_cfg["gradflow"]["stop_at"]

    for ep in pbar:
        model.train()
        batch = exp.sample_batch(n_f=n_f, n_b=n_b, n_0=n_0)

        loss_f = exp.pde_residual_loss(model, batch).mean() if batch.get("X_f") is not None else torch.tensor(0., device=device)
        loss_b = exp.boundary_loss(model, batch).mean()     if batch.get("X_b") is not None else torch.tensor(0., device=device)
        loss_0 = exp.initial_loss(model, batch).mean()      if batch.get("X_0") is not None else torch.tensor(0., device=device)

        loss_f_s = loss_f.mean() if torch.is_tensor(loss_f) and loss_f.dim() > 0 else loss_f
        loss_b_s = loss_b.mean() if torch.is_tensor(loss_b) and loss_b.dim() > 0 else loss_b
        loss_0_s = loss_0.mean() if torch.is_tensor(loss_0) and loss_0.dim() > 0 else loss_0

        if ep <= gf_stop:
            gf.collect({"res": loss_f_s, "bc": loss_b_s, "ic": loss_0_s})

        total_loss = w_f*loss_f + w_b*loss_b + w_ic*loss_0

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # Log
        log_dict = {
            "loss/total": total_loss.item(),
            "loss/f": loss_f.item(),
            "loss/b": loss_b.item(),
            "loss/ic": loss_0.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": ep
        }
        wandb_log(log_dict)
        pbar.set_postfix({k: f"{v:.3e}" for k,v in log_dict.items() if "loss" in k})

        # Simple validation metric (relative L2 on a fixed grid)
        best_path = os.path.join(out_dir, "best.pt")
        if ep % eval_every == 0 or ep == epochs-1:
            with torch.no_grad():
                rel_l2 = exp.relative_l2_on_grid(model, base_cfg["eval"]["grid"])
            wandb_log({"eval/rel_l2": rel_l2, "epoch": ep})

            if rel_l2 < (best_metric - es_cfg.get("min_delta", 0.0)):
                best_metric = rel_l2
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
                torch.save({k:v.detach().cpu() for k,v in best_state.items()}, best_path)

            if early and early.step(rel_l2):
                print(f"\n[EarlyStopping] Stopping at epoch {ep}. Best rel_l2={best_metric:.3e}")
                break

    gf.save()
    if gf.plot_enabled:
        saved = gf.save_plots()
        for p in saved:
            print(f"[gradflow] saved plot: {p}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation & plots
    model.eval()
    figs = exp.plot_final(model, base_cfg["eval"]["grid"], out_dir)
    for name, path in figs.items():
        wandb_log({f"fig/{name}": wandb.Image(path)})

    wandb_finish()
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--common_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--exp_config", required=True)
    args = parser.parse_args()
    main(args)