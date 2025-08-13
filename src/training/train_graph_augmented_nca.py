import os
import re
import glob
import json
import time
import random
from datetime import datetime
import sys
import signal  # <-- NEW

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.config import load_config
from utils.image import load_single_target_image
from training.pool import SamplePool
from utils.nca_init import make_seed
from utils.utility_functions import count_parameters
from utils.visualize import save_comparison

# Graph-augmented NCA (your class in src/modules/ncagraph.py)
from modules.ncagraph import NeuralCAGraph

# ---------------------------------------------------------------------------
# Masked loss: supervise where TARGET is alive + tiny area penalty on alpha
# ---------------------------------------------------------------------------
def masked_loss(pred, target, alpha_thr: float = 0.2, lam_area: float = 5e-5):
    """
    pred, target: [B, 4, H, W]
    Loss is MSE masked by TARGET alpha > alpha_thr, plus a tiny 'area' penalty
    on predicted alpha to discourage sprawl.
    Returns: per-sample vector [B]
    """
    target_mask = (target[:, 3:4] > alpha_thr).float()   # [B,1,H,W]
    mse = ((pred - target) ** 2) * target_mask
    denom = target_mask.sum(dim=(1, 2, 3)) + 1e-8
    per_sample = mse.sum(dim=(1, 2, 3)) / denom
    area_pen = lam_area * pred[:, 3:4].mean(dim=(1, 2, 3))
    return per_sample + area_pen


def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1, 2, 0).numpy().clip(0, 1)
    plt.imsave(filename, img)


def main():
    start_wall = time.time()
    config = load_config("configs/config.json")

    # ---- IO dirs (separate tree for graph runs) ----
    target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "graphaug_nca", "train_inter_loss", target_name)
    results_dir = os.path.join(base_dir, "images")
    inter_dir   = os.path.join(base_dir, "intermediate")
    ckpt_dir    = os.path.join(base_dir, "checkpoints")
    tb_dir      = os.path.join(base_dir, "tb_logs")
    logs_dir    = os.path.join(base_dir, "logs")
    for d in [results_dir, inter_dir, ckpt_dir, tb_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    device = config["misc"]["device"]
    torch.manual_seed(config["misc"]["seed"])

    # ---- Data ----
    target   = load_single_target_image(config).to(device)  # [4,H,W]
    img_size = int(config["data"]["img_size"])
    n_ch     = int(config["model"]["n_channels"])

    writer = SummaryWriter(log_dir=tb_dir)

    # ---- Seed: alpha=1 at center; tiny noise on hidden channels only ----
    def seed_fn(batch_size=1):
        g = torch.zeros(batch_size, n_ch, img_size, img_size, device=device)
        cy = img_size // 2; cx = img_size // 2
        g[:, 3:4, cy, cx] = 1.0
        if n_ch > 4:
            g[:, 4:, cy, cx] = 0.01 * torch.randn_like(g[:, 4:, cy, cx])
        return g
    
    
    # ---- Model (Graph-aug NCA) ----
    gcfg = config.get("graph_augmentation", {})
    model = NeuralCAGraph(
        n_channels=n_ch,
        update_hidden=int(config["model"]["update_mlp"]["hidden_dim"]),
        img_size=img_size,
        update_gain=float(config["model"].get("update_gain", 0.1)),
        alpha_thr=float(config["model"].get("alpha_thr", 0.1)),
        use_groupnorm=bool(config["model"].get("use_groupnorm", True)),
        # Graph knobs
        message_gain=float(gcfg.get("message_gain", 0.5)),
        hidden_only=bool(gcfg.get("hidden_only", True)),
        graph_d_model=int(gcfg.get("d_model", 16)),
        graph_attention_radius=int(gcfg.get("attention_radius", 4)),
        graph_num_neighbors=int(gcfg.get("num_neighbors", 8)),
        graph_gating_hidden=int(gcfg.get("gating_hidden", 32)),
        device=device,
    ).to(device)
    model.train()

    # Temporal sparsity for mid-range messages
    msg_rate  = float(gcfg.get("message_rate", 1.0))
    msg_every = int(gcfg.get("message_every", 1))  # 1 = every step

    # ---- Optimizer / Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    scheduler_cfg = config["training"].get("scheduler")
    scheduler = None
    if isinstance(scheduler_cfg, dict):
        t = scheduler_cfg.get("type", "").lower()
        if t == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_cfg.get("step_size", 50)),
                gamma=float(scheduler_cfg.get("gamma", 0.7)),
            )
        elif t == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("t_max", 200)),
                eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
            )

    # ---- Pool ----
    pool = SamplePool(int(config["training"]["pool_size"]), seed_fn, device=device)

    # ---- Training knobs ----
    total_epochs    = int(config["training"]["num_epochs"])
    steps_per_epoch = int(config["training"]["steps_per_epoch"])
    batch_size      = int(config["training"]["batch_size"])
    short_min       = int(config["training"]["nca_steps_min"])
    short_max       = int(config["training"]["nca_steps_max"])
    long_prob       = float(config["training"].get("long_rollout_prob", 0.25))
    long_min        = int(config["training"].get("long_rollout_steps_min", 200))
    long_max        = int(config["training"].get("long_rollout_steps_max", 400))
    fr_min          = float(config["training"].get("fire_rate_min", 0.5))
    fr_max          = float(config["training"].get("fire_rate_max", 1.0))

    log_interval    = int(config["logging"]["log_interval"])
    visualize_every = int(config["logging"]["visualize_interval"])
    ckpt_interval   = int(config["logging"].get("checkpoint_interval_epochs", 25))

    # Resets
    reset_worst_prob   = float(config["training"].get("reset_worst_prob", 0.10))
    random_reseed_prob = float(config["training"].get("random_reseed_prob", 0.05))

    # Stability phase
    stab_enabled   = bool(config["training"].get("stability_enabled", True))
    stab_K         = int(config["training"].get("stability_K", 24))
    stab_thresh    = float(config["training"].get("stability_threshold", 0.01))
    stab_weight    = float(config["training"].get("stability_weight", 0.5))

    # Loss knobs
    loss_alpha_thr = float(config["training"].get("loss_alpha_thr", 0.2))
    loss_lam_area  = float(config["training"].get("loss_lam_area", 5e-5))

    # ---- Resume (strict=False for safety across versions) ----
    def extract_epoch_num(name):
        m = re.search(r'epoch(\d+)', name)
        return int(m.group(1)) if m else -1

    ckpt_files = glob.glob(os.path.join(ckpt_dir, "nca_epoch*.pt"))
    if ckpt_files:
        ckpt_files.sort(key=extract_epoch_num)
        last_ckpt = ckpt_files[-1]
        checkpoint = torch.load(last_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing:   print(f"[resume] missing model keys: {missing}")
        if unexpected:print(f"[resume] unexpected model keys: {unexpected}")
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[warn] optimizer state not compatible, reinitializing: {e}")
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            except Exception as e:
                print(f"[warn] scheduler state not compatible, reinit: {e}")
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming (fine-tune) from {last_ckpt} (epoch {checkpoint['epoch']})")
    else:
        start_epoch = 1
        print("Starting training from scratch.")

    # ---- Bookkeeping ----
    n_params = count_parameters(model)
    print(f"Params (graph NCA): {n_params}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")

    epoch_losses, pixel_scores, ssim_scores, psnr_scores = [], [], [], []

    # ---------- NEW: graceful termination handler (save last checkpoint) ----------
    terminate = {"flag": False}

    def _save_ckpt(tag: str, epoch_val: int, global_step_val: int):
        ckpt_path = os.path.join(ckpt_dir, f"nca_{tag}.pt")
        torch.save({
            "epoch": epoch_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "param_count": n_params,
            "global_step": global_step_val,
        }, ckpt_path)
        print(f"[ckpt] saved {ckpt_path}")

    def _handle_term(signum, frame):
        # Just set a flag; saving will happen safely in the training loop
        print(f"[signal] Received {signum}. Will save a LAST checkpoint and exit cleanly...")
        terminate["flag"] = True

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT,  _handle_term)
    # ------------------------------------------------------------------------------

    try:
        for epoch in trange(start_epoch, total_epochs + 1, desc="Epochs"):
            avg_loss = 0.0
            epoch_pixel, epoch_ssim, epoch_psnr = [], [], []

            for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
                idx, batch = pool.sample(batch_size)
                state = batch.to(device)

                # Pick rollout regime
                if random.random() < long_prob:
                    steps_lo, steps_hi = long_min, long_max
                else:
                    steps_lo, steps_hi = short_min, short_max

                nca_steps = torch.randint(steps_lo, steps_hi + 1, (batch_size,), device=device)
                max_steps = int(nca_steps.max().item())

                # Rollout with random fire-rate & temporally sparse messages
                base_gain = float(getattr(model, "message_gain", 0.5))

                for t in range(max_steps):
                    mask = (nca_steps > t)
                    if not mask.any():
                        continue

                    # fire-rate for this internal step
                    fr = float(torch.empty(1, device=device).uniform_(fr_min, fr_max).item())

                    # decide whether to use graph this step
                    use_graph = True
                    if msg_every > 1:
                        use_graph = (t % msg_every == 0)
                    elif msg_rate < 1.0:
                        use_graph = (random.random() < msg_rate)

                    if hasattr(model, "message_gain"):
                        model.message_gain = base_gain if use_graph else 0.0

                    state[mask] = model(state[mask], fire_rate=fr)

                # Restore original gain
                if hasattr(model, "message_gain"):
                    model.message_gain = base_gain

                # Target batch
                target_expand = target.unsqueeze(0).expand_as(state[:, :4])

                # Primary loss
                per_sample = masked_loss(state[:, :4], target_expand,
                                         alpha_thr=loss_alpha_thr, lam_area=loss_lam_area)
                loss = per_sample.mean()

                # Stability phase
                if stab_enabled:
                    with torch.no_grad():
                        close_mask = (per_sample < stab_thresh)
                    if close_mask.any():
                        state_stab = state.clone()
                        base_gain = float(getattr(model, "message_gain", 0.5))
                        for k in range(stab_K):
                            frs = float(torch.empty(1, device=device).uniform_(fr_min, fr_max).item())
                            use_graph = True
                            if msg_every > 1:
                                use_graph = (k % msg_every == 0)
                            elif msg_rate < 1.0:
                                use_graph = (random.random() < msg_rate)
                            if hasattr(model, "message_gain"):
                                model.message_gain = base_gain if use_graph else 0.0
                            state_stab[close_mask] = model(state_stab[close_mask], fire_rate=frs)
                        if hasattr(model, "message_gain"):
                            model.message_gain = base_gain
                        stab = F.mse_loss(state_stab[close_mask, :4], target_expand[close_mask])
                        loss = loss + stab_weight * stab

                # Decide resets (apply after step on detached copy)
                n_reset = int(reset_worst_prob * batch_size)
                worst_indices = torch.topk(per_sample, n_reset).indices if n_reset > 0 else None
                do_random_reseed = (random.random() < random_reseed_prob)
                rand_idx = torch.randint(0, batch_size, (1,), device=device).item() if do_random_reseed else None

                # Optimize
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                # Pool replace on detached copy
                state_for_pool = state.detach()
                if (worst_indices is not None) or do_random_reseed:
                    state_for_pool = state_for_pool.clone()
                    if worst_indices is not None:
                        state_for_pool[worst_indices] = seed_fn(len(worst_indices)).detach()
                    if do_random_reseed:
                        state_for_pool[rand_idx:rand_idx+1] = seed_fn(1).detach()
                pool.replace(idx, state_for_pool)

                # Metrics (on current state)
                pred = state[:, :4]
                pred_img = pred[0].detach().cpu().numpy()
                target_img = target.cpu().numpy()
                diff = np.abs(pred_img[:4] - target_img[:4])
                pixel_perfect = (diff < 0.05).all(axis=0).mean()
                epoch_pixel.append(float(pixel_perfect))

                pred_np = pred[0, :3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                target_np = target[:3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                epoch_ssim.append(ssim(pred_np, target_np, data_range=1.0, channel_axis=-1))
                epoch_psnr.append(psnr(pred_np, target_np, data_range=1.0))

                avg_loss += loss.item()

                # Logging
                global_step = (epoch - 1) * steps_per_epoch + step
                writer.add_scalar('Loss/train', loss.item(), global_step)

                if (step + 1) % visualize_every == 0:
                    img = pred[0, :3].detach().cpu().clamp(0, 1)
                    writer.add_image('Predicted/sample', img, global_step)
                    save_path0 = os.path.join(results_dir, f"epoch{epoch}_step{step+1}_sample0.png")
                    save_grid_as_image(state[0], save_path0)
                    save_comparison(target, pred[0], f"{epoch}_step{step+1}_sample0", results_dir, upscale=4)

                if (step + 1) % log_interval == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                          f"Loss: {loss.item():.5f}")

                # --------- NEW: if SLURM told us to stop, save & exit now ----------
                if terminate["flag"]:
                    _save_ckpt(f"ep{epoch}_step{step+1}_last", epoch, global_step)
                    writer.flush(); writer.close()
                    print("[signal] Last checkpoint saved. Exiting.")
                    sys.exit(0)
                # ------------------------------------------------------------------

            # End of epoch
            avg_loss /= steps_per_epoch
            epoch_losses.append(avg_loss)
            pixel_scores.append(float(np.mean(epoch_pixel)))
            ssim_scores.append(float(np.mean(epoch_ssim)))
            psnr_scores.append(float(np.mean(epoch_psnr)))

            # Log row
            with open(os.path.join(logs_dir, "training_log.jsonl"), "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "avg_loss": float(avg_loss),
                    "pixel_perfection": float(np.mean(epoch_pixel)),
                    "ssim": float(np.mean(epoch_ssim)),
                    "psnr": float(np.mean(epoch_psnr)),
                    "timestamp": datetime.now().isoformat()
                }) + "\n")

            writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
                  f"Average loss: {avg_loss:.6f}")

            if scheduler is not None:
                scheduler.step()

            # Checkpoint
            if epoch % ckpt_interval == 0 or epoch == total_epochs:
                ckpt_path = os.path.join(ckpt_dir, f"nca_epoch{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "config": config,
                }, ckpt_path)
                print(f"Model checkpoint saved to {ckpt_path}")

            # Also honor termination right after epoch boundary
            if terminate["flag"]:
                global_step = epoch * steps_per_epoch  # end-of-epoch step index
                _save_ckpt(f"epoch{epoch}_last", epoch, global_step)
                writer.flush(); writer.close()
                print("[signal] Last checkpoint saved at epoch boundary. Exiting.")
                sys.exit(0)

    except Exception as e:
        # Optional: write a last-ditch checkpoint on unexpected crash
        try:
            epoch_safe = locals().get("epoch", -1)
            step_safe  = locals().get("step", -1)
            global_step = (max(epoch_safe, 1) - 1) * steps_per_epoch + max(step_safe, 0)
            _save_ckpt(f"crash_ep{epoch_safe}_step{step_safe}", max(epoch_safe, 1), global_step)
            print(f"[crash] Saved emergency checkpoint due to: {e}")
        finally:
            raise

    # Summary JSON
    writer.flush(); writer.close()
    log_data = {
        "training_time_minutes": (time.time() - start_wall) / 60.0,
        "config": config,
        "parameter_count": n_params,
        "seed": config["misc"].get("seed"),
        "initial_loss": float(epoch_losses[0]) if epoch_losses else None,
        "final_loss": float(avg_loss) if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "average_pixel_perfection": float(np.mean(pixel_scores)) if pixel_scores else None,
        "pixel_perfection_per_epoch": pixel_scores,
        "average_ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
        "ssim_per_epoch": ssim_scores,
        "average_psnr": float(np.mean(psnr_scores)) if psnr_scores else None,
        "psnr_per_epoch": psnr_scores
    }
    with open(os.path.join(logs_dir, "training_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved training log to {os.path.join(logs_dir, 'training_log.json')}")


if __name__ == "__main__":
    main()
