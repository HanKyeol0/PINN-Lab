import os, time, yaml, argparse, sys
import torch
import wandb
import csv
from tqdm import trange
from pinnlab.registry import get_model, get_experiment
from pinnlab.utils.seed import seed_everything
from pinnlab.utils.early_stopping import EarlyStopping
from pinnlab.utils.plotting import save_plots_1d, save_plots_2d
from pinnlab.utils.wandb_utils import setup_wandb, wandb_log, wandb_finish
from pinnlab.utils.loss_balancer import BalancerConfig, make_loss_balancer
from pinnlab.utils.plotting import plot_weights_over_time

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

    if exp_cfg.get("device"):
        base_cfg["device"] = exp_cfg["device"]
    device = torch.device(base_cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    
    exp = get_experiment(args.experiment_name)(exp_cfg, device)
    model = get_model(args.model_name)(model_cfg).to(device)
    
    # Logging dir
    tag = exp_cfg.get("tag", None)
    if tag:
        file_name = f"{args.experiment_name}_{args.model_name}_{tag}"
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{args.experiment_name}_{args.model_name}_{ts}"
    
    out_dir = os.path.join(base_cfg["log"]["out_dir"], args.experiment_name, file_name)
    os.makedirs(out_dir, exist_ok=True)

    _save_yaml(os.path.join(out_dir, "config.yaml"), {
        "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
    })
    
    # Loss balancer
    use_loss_balancer = base_cfg["train"]["loss_balancer"].get("use_loss_balancer", False)
    if use_loss_balancer:
        lb_cfg_dict = base_cfg["train"].get("loss_balancer", {})  # {'kind': 'dwa', 'terms': ['res','bc','ic'], ...}
        lb_cfg = BalancerConfig(**lb_cfg_dict)
        if not lb_cfg.terms:
            lb_cfg.terms = list(base_cfg["train"]["loss_weights"].keys()) # ["res", "bc", "ic", "data"]

        balancer = make_loss_balancer(lb_cfg)

    weights_csv = os.path.join(out_dir, "loss_weights.csv")
    weights_terms = None  # will infer on first log

    # Optimizer
    params = list(model.parameters())
    if use_loss_balancer:
       params += list(balancer.extra_params())  # no-op for other schemes

    opt_cfg = base_cfg["train"]["optimizer"]
    if opt_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    else:
        raise ValueError("Only Adam is wired in, add more in train.py.")

    # create file with header late (when we know the terms)
    def _ensure_weights_header(terms):
        nonlocal weights_terms
        if weights_terms is None:
            weights_terms = list(terms)
            with open(weights_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + weights_terms)

    # WandB
    if base_cfg["log"]["wandb"]["enabled"]:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(project = base_cfg["log"]["wandb"]["project"],
                   name = file_name)
        run = setup_wandb(base_cfg["log"]["wandb"], args, out_dir, config={
            "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
        })

    epochs = base_cfg["train"]["epochs"]
    eval_every = int(base_cfg.get("eval").get("every", 100))

    # Early stopping
    es_cfg = base_cfg["train"]["early_stopping"]
    early = EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"], eval_every=eval_every) if es_cfg["enabled"] else None
    best_state = None
    best_metric = float("inf")

    w_res = base_cfg["train"]["loss_weights"]["res"]
    w_bc = base_cfg["train"]["loss_weights"]["bc"]
    w_ic = base_cfg["train"]["loss_weights"]["ic"]
    w_data = base_cfg["train"]["loss_weights"]["data"]

    n_f = exp_cfg.get("batch", {}).get("n_f", base_cfg["train"]["batch"]["n_f"])
    n_b = exp_cfg.get("batch", {}).get("n_b", base_cfg["train"]["batch"]["n_b"])
    n_0 = exp_cfg.get("batch", {}).get("n_0", base_cfg["train"]["batch"]["n_0"])
    
    # Make video
    enable_video = exp_cfg.get("video", {}).get("enabled", False)
    make_video_every = exp_cfg.get("video", {}).get("every", eval_every)

    use_tty = sys.stdout.isatty()
    pbar = trange(
        epochs,
        desc="Training",
        ncols=120,
        dynamic_ncols=True,
        leave=False,          # don't leave old bars behind
        disable=not use_tty,  # if output is piped, avoid multiline spam
    )

    # Training loop
    print("training started")
    training_start_time = time.time()
    global_step = 0
    for ep in pbar:
        model.train()
        batch = exp.sample_batch(n_f=n_f, n_b=n_b, n_0=n_0)

        loss_res = exp.pde_residual_loss(model, batch).mean() if batch.get("X_f") is not None else torch.tensor(0., device=device)
        # loss_bc = exp.boundary_loss(model, batch).mean()      if batch.get("X_b") is not None else torch.tensor(0., device=device)
        # loss_ic = exp.initial_loss(model, batch).mean()       if batch.get("X_0") is not None else torch.tensor(0., device=device)
        loss_data = exp.data_loss(model, batch).mean()        if batch.get("X_d") is not None else torch.tensor(0., device=device)
        
        loss_res_s = loss_res.mean() if torch.is_tensor(loss_res) and loss_res.dim() > 0 else loss_res # scalar
        # loss_bc_s = loss_bc.mean() if torch.is_tensor(loss_bc) and loss_bc.dim() > 0 else loss_bc
        # loss_ic_s = loss_ic.mean() if torch.is_tensor(loss_ic) and loss_ic.dim() > 0 else loss_ic
        loss_data_s = loss_data.mean() if torch.is_tensor(loss_data) and loss_data.dim() > 0 else loss_data

        losses = {
            "res": loss_res,     # PDE residual term
            # **({"bc": loss_bc} if "loss_bc" in locals() else {}),
            # **({"ic": loss_ic} if "loss_ic" in locals() else {}),
            **({"data": loss_data} if "loss_data" in locals() else {}),
        }

        if not use_loss_balancer:
            total_loss = w_res*loss_res + w_data*loss_data
            # total_loss = w_res*loss_res + w_bc*loss_bc + w_ic*loss_ic + w_data*loss_data
            s = (w_res + w_bc + w_ic + w_data) or 1.0
            w_now = {"res": w_res/s, "bc": w_bc/s, "ic": w_ic/s, "data": w_data/s}
        else:
            total_loss, w_dict, aux = balancer(losses, step=global_step, model=model)
            w_now = {k.split("/", 1)[1]: float(v) for k, v in w_dict.items()}

        # write one row per epoch/step
        _ensure_weights_header(w_now.keys())
        with open(weights_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([global_step] + [w_now[t] for t in weights_terms])

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # Log
        it_per_sec = pbar.format_dict.get("rate", None)
        elapsed_s  = pbar.format_dict.get("elapsed", None)
        gpu_now = {
            "gpu/mem_alloc_mb": float(torch.cuda.memory_allocated(device)) / (1024**2),
            "gpu/mem_reserved_mb": float(torch.cuda.memory_reserved(device)) / (1024**2),
        }
        log_payload = {
            "loss/total": float(total_loss.detach().cpu()),
            "loss/res": float(loss_res_s.detach().cpu()),
            # "loss/bc": float(loss_bc_s.detach().cpu()),
            # "loss/ic": float(loss_ic_s.detach().cpu()),
            "loss/data": float(loss_data_s.detach().cpu()),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": ep,
            "perf/it_per_sec_tqdm": it_per_sec if it_per_sec is not None else 0.0,
            "perf/elapsed_sec": elapsed_s if elapsed_s is not None else 0.0,
            **gpu_now,
        }
        wandb_log(log_payload, commit=True)
        pbar.set_postfix({k: f"{v:.3e}" for k,v in log_payload.items() if "loss" in k})
        global_step += 1

        # Simple validation metric (relative L2 on a fixed grid)
        if (ep % eval_every == 0 or ep == epochs-1) and (ep > 0):
            with torch.no_grad():
                rel_l2 = exp.relative_l2_on_grid(model, base_cfg["eval"]["grid"])
            wandb_log({"eval/rel_l2": rel_l2, "epoch": ep})

            best_path = os.path.join(out_dir, "best.pt")
            if rel_l2 < (best_metric - es_cfg.get("min_delta", 0.0)):
                best_metric = rel_l2
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
                torch.save({k:v.detach().cpu() for k,v in best_state.items()}, best_path)

            if early and early.step(rel_l2):
                print(f"\n[EarlyStopping] Stopping at epoch {ep}. Best rel_l2={best_metric:.3e}")
                break
            
        if enable_video and (ep % make_video_every == 0 and ep > 0):
            vid_grid = exp_cfg.get("video", {}).get("grid", base_cfg["eval"]["grid"])
            fps      = exp_cfg.get("video", {}).get("fps", 10)
            out_fmt  = exp_cfg.get("video", {}).get("format", "mp4")  # "mp4" or "gif"
            vid_path = exp.make_video(
                model, vid_grid, out_dir, fps=fps,
                filename=f"eval_ep{ep}.{out_fmt}"
            )

    training_end_time = time.time()
    
    final_perf = {
        "perf/total_time_sec": training_end_time - training_start_time,
        "gpu/peak_mem_alloc_mb": float(torch.cuda.max_memory_allocated(device)) / (1024**2),
        "gpu/peak_mem_reserved_mb": float(torch.cuda.max_memory_reserved(device)) / (1024**2),
    }

    if enable_video:
        vid_grid = exp_cfg.get("video", {}).get("grid", base_cfg["eval"]["grid"])
        fps      = exp_cfg.get("video", {}).get("fps", 10)
        out_fmt  = exp_cfg.get("video", {}).get("format", "mp4")  # "mp4" or "gif"
        vid_path = exp.make_video(
            model, vid_grid, out_dir,
            fps=fps, filename=f"final_evolution.{out_fmt}"
        )
        wandb_log({"video/evolution": wandb.Video(vid_path, format=out_fmt)})
        
        base, ext = os.path.splitext(os.path.basename(vid_path))
        noise_true = os.path.join(out_dir, f"{base}_noise_true{ext}")
        noise_ebm  = os.path.join(out_dir, f"{base}_noise_ebm{ext}")
        if os.path.exists(noise_true):
            wandb_log({"video/noise_true": wandb.Video(noise_true, format=out_fmt)})
        if os.path.exists(noise_ebm):
            wandb_log({"video/noise_ebm": wandb.Video(noise_ebm, format=out_fmt)})    
    
    wandb_log(final_perf)

    weights_png = os.path.join(out_dir, "loss_weights.png")
    plot_weights_over_time(weights_csv, weights_png)
    print(f"[weights] saved: {weights_csv}")
    print(f"[weights] plot : {weights_png}")

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