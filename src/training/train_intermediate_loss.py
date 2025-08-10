# src/training/train_intermediate_loss.py
import os
import re
import glob
import json
import time
import random
from datetime import datetime

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
from modules.nca import NeuralCA
from utils.visualize import save_comparison
from utils.utility_functions import count_parameters

# Uncomment while debugging to pinpoint any autograd issue
# torch.autograd.set_detect_anomaly(True)


# -----------------------------------------------------------------------------
# Masked loss: supervise where TARGET is alive; add tiny area penalty
# -----------------------------------------------------------------------------
def masked_loss(pred, target, alpha_thr: float = 0.2, lam_area: float = 5e-5):
    """
    pred, target: [B, 4, H, W] (RGBA)
    - Mask by TARGET alpha > alpha_thr, not by pred alpha (prevents sprawl)
    - Add tiny area penalty on predicted alpha to discourage flooding
    Returns: per-sample vector [B]
    """
    target_mask = (target[:, 3:4] > alpha_thr).float()  # [B,1,H,W]
    mse = ((pred - target) ** 2) * target_mask
    denom = target_mask.sum(dim=(1, 2, 3)) + 1e-8
    per_sample = mse.sum(dim=(1, 2, 3)) / denom

    # Very small pressure to keep alpha compact (acts everywhere)
    area_pen = lam_area * pred[:, 3:4].mean(dim=(1, 2, 3))
    return per_sample + area_pen


# -----------------------------------------------------------------------------
# Utilities: save a single grid image for quick visual checks
# -----------------------------------------------------------------------------
def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1, 2, 0).numpy().clip(0, 1)
    plt.imsave(filename, img)


def save_debug_canvas(state, path, idx=0):
    with torch.no_grad():
        img = state[idx, :3].detach().cpu().clamp(0, 1)
        alpha = state[idx, 3].detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img.permute(1, 2, 0).numpy())
        axs[0].set_title('RGB')
        axs[1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
        axs[1].set_title('Alpha channel')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------
def main():
    start_wall = time.time()
    config = load_config("configs/config.json")

    active_target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "classic_nca", "train_inter_loss", active_target_name)
    results_dir = os.path.join(base_dir, "images")
    inter_dir = os.path.join(base_dir, "intermediate")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    config["logging"]["results_dir"] = results_dir
    config["logging"]["checkpoint_dir"] = ckpt_dir

    device = config["misc"]["device"]
    torch.manual_seed(config["misc"]["seed"])

    # Data / target
    target = load_single_target_image(config).to(device)  # [4, H, W]
    img_size = config["data"]["img_size"]
    n_channels = config["model"]["n_channels"]

    # TensorBoard
    writer = SummaryWriter(log_dir=tb_dir)

    # Cooler seed: only alpha=1.0 at center; tiny hidden-state noise
    def seed_fn(batch_size=1):
        g = torch.zeros(batch_size, n_channels, img_size, img_size, device=device)
        cy = img_size // 2
        cx = img_size // 2
        g[:, 3:4, cy, cx] = 1.0
        if n_channels > 4:
            g[:, 4:, cy, cx] = 0.01 * torch.randn_like(g[:, 4:, cy, cx])
        return g

    # Model (drop-in NeuralCA with bounded dx & GroupNorm-on-dx)
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config["model"]["update_mlp"]["hidden_dim"],
        img_size=img_size,
        update_gain=0.1,            # small step size 0.05–0.2
        alpha_thr=0.1,              # alive threshold in the CA dynamics
        use_groupnorm=True,
        device=device
    ).to(device)
    model.train()

    # Optimizer (Adam + small weight decay)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # (Optional) LR scheduler (controlled via config)
    scheduler_cfg = config["training"].get("scheduler")
    scheduler = None
    if isinstance(scheduler_cfg, dict):
        if scheduler_cfg.get("type", "").lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_cfg.get("step_size", 50)),
                gamma=float(scheduler_cfg.get("gamma", 0.7)),
            )
        elif scheduler_cfg.get("type", "").lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("t_max", 200)),
                eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
            )

    # Pool
    pool = SamplePool(config["training"]["pool_size"], seed_fn, device=device)

    # Training params
    total_epochs = config["training"]["num_epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]
    batch_size = config["training"]["batch_size"]
    short_min = config["training"]["nca_steps_min"]   # 48
    short_max = config["training"]["nca_steps_max"]   # 80
    long_prob = 0.25                                   # 25% long rollouts
    long_min, long_max = 200, 400                      # long horizons
    log_interval = config["logging"]["log_interval"]
    visualize_interval = config["logging"]["visualize_interval"]
    ckpt_interval = config["logging"].get("checkpoint_interval_epochs", 25)

    # Reset policy (applied AFTER backward/step)
    reset_worst_prob = 0.10    # reset top-10% losses in batch
    random_reseed_prob = 0.05  # plus small random reseed to maintain age diversity

    # Metrics storage
    epoch_losses, pixel_scores, ssim_scores, psnr_scores = [], [], [], []

    # Resume if checkpoints exist
    def extract_epoch_num(ckpt_filename):
        m = re.search(r'epoch(\d+)', ckpt_filename)
        return int(m.group(1)) if m else -1

    ckpt_files = glob.glob(os.path.join(ckpt_dir, "nca_epoch*.pt"))
    if ckpt_files:
        ckpt_files = sorted(ckpt_files, key=extract_epoch_num)
        last_ckpt = ckpt_files[-1]
        checkpoint = torch.load(last_ckpt, map_location=device)
    
        # 1) Model: allow missing (new GN) / unexpected keys (name changes)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing:   print(f"[resume] missing keys (initialized fresh): {missing}")
        if unexpected:print(f"[resume] unexpected keys (ignored): {unexpected}")
    
        # 2) Optimizer: try to load, else re-init fresh
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[warn] optimizer state not compatible, reinitializing: {e}")
    
        # 3) Scheduler: only if present & compatible
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            except Exception as e:
                print(f"[warn] scheduler state not compatible, reinitializing: {e}")
    
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming (fine-tune) from {last_ckpt} (epoch {checkpoint['epoch']})")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6g}")
    else:
        start_epoch = 1
        print("Starting training from scratch.")


    # Parameter count
    nca_param_count = count_parameters(model)
    print(f"Params: {nca_param_count}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")

    for epoch in trange(start_epoch, total_epochs + 1, desc="Epochs"):
        avg_loss = 0.0
        epoch_pixel, epoch_ssim, epoch_psnr = [], [], []

        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            # Sample batch from pool
            idx, batch = pool.sample(batch_size)
            state = batch.to(device)

            # Damage training (optional) — currently commented
            # if torch.rand(1) < 0.30:
            #     s = config["training"]["damage_patch_size"]  # e.g., 10
            #     y0 = torch.randint(0, img_size - s + 1, (1,), device=device).item()
            #     x0 = torch.randint(0, img_size - s + 1, (1,), device=device).item()
            #     state[:, :, y0:y0 + s, x0:x0 + s] = 0.0

            # Choose rollout regime (short vs long)
            if random.random() < long_prob:
                steps_lo, steps_hi = long_min, long_max
            else:
                steps_lo, steps_hi = short_min, short_max

            nca_steps = torch.randint(steps_lo, steps_hi + 1, (batch_size,), device=device)
            max_steps = int(nca_steps.max().item())

            # Rollout with randomized fire-rate per step (asynchronous-to-synchronous mix)
            for t in range(max_steps):
                mask = (nca_steps > t)
                if mask.any():
                    fire_rate = float(torch.empty(1, device=device).uniform_(0.5, 1.0).item())
                    state[mask] = model(state[mask], fire_rate=fire_rate)

            # Build target batch
            target_expand = target.unsqueeze(0).expand_as(state[:, :4])

            # Primary loss (per-sample), then mean
            per_sample = masked_loss(state[:, :4], target_expand, alpha_thr=0.2, lam_area=5e-5)
            loss = per_sample.mean()

            # Stability phase: for samples already near target, roll extra K steps & penalize drift
            with torch.no_grad():
                close_mask = (per_sample < 0.01)  # tune threshold

            if close_mask.any():
                state_stab = state.clone()
                K = 24
                for _ in range(K):
                    fr = float(torch.empty(1, device=device).uniform_(0.5, 1.0).item())
                    state_stab[close_mask] = model(state_stab[close_mask], fire_rate=fr)
                stab = F.mse_loss(state_stab[close_mask, :4], target_expand[close_mask])
                loss = loss + 0.5 * stab  # weight 0.3–1.0

            # Decide which indices to reset/reseed, but DO NOT modify `state` yet
            n_reset = int(reset_worst_prob * batch_size)
            worst_indices = None
            if n_reset > 0:
                worst_indices = torch.topk(per_sample, n_reset).indices

            do_random_reseed = (random.random() < random_reseed_prob)
            rand_idx = None
            if do_random_reseed:
                rand_idx = torch.randint(0, batch_size, (1,), device=device).item()

            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Prepare detached copy for the pool and apply resets/reseeds SAFELY
            state_for_pool = state.detach()
            # Only clone if we actually need to modify it
            if (worst_indices is not None) or do_random_reseed:
                state_for_pool = state_for_pool.clone()
                if worst_indices is not None:
                    state_for_pool[worst_indices] = seed_fn(len(worst_indices)).detach()
                if do_random_reseed:
                    state_for_pool[rand_idx:rand_idx+1] = seed_fn(1).detach()

            # Return updated states to pool (detached)
            pool.replace(idx, state_for_pool)

            # --- Logging / metrics (computed on the *pre-reset* state) ---
            pred = state[:, :4]
            pred_img = pred[0].detach().cpu().numpy()
            target_img = target.cpu().numpy()
            diff = np.abs(pred_img[:4] - target_img[:4])
            pixel_perfect = (diff < 0.05).all(axis=0).mean()
            epoch_pixel.append(float(pixel_perfect))

            pred_np = pred[0, :3].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            target_np = target[:3].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            epoch_ssim.append(ssim(pred_np, target_np, data_range=1.0, channel_axis=-1))
            epoch_psnr.append(psnr(pred_np, target_np, data_range=1.0))

            avg_loss += loss.item()

            global_step = (epoch - 1) * steps_per_epoch + step
            writer.add_scalar('Loss/train', loss.item(), global_step)

            if (step + 1) % visualize_interval == 0:
                img = pred[0, :3].detach().cpu().clamp(0, 1)
                writer.add_image('Predicted/sample', img, global_step)
                save_path0 = os.path.join(results_dir, f"epoch{epoch}_step{step+1}_sample0.png")
                save_grid_as_image(state[0], save_path0)
                save_comparison(target, pred[0], f"{epoch}_step{step+1}_sample0", results_dir, upscale=4)

            if (step + 1) % log_interval == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                      f"Loss: {loss.item():.5f}")

        # End of epoch: averages
        avg_loss /= steps_per_epoch
        epoch_losses.append(avg_loss)
        pixel_scores.append(float(np.mean(epoch_pixel)))
        ssim_scores.append(float(np.mean(epoch_ssim)))
        psnr_scores.append(float(np.mean(epoch_psnr)))

        # write per-epoch log row
        epoch_log = {
            "epoch": epoch,
            "avg_loss": float(avg_loss),
            "pixel_perfection": float(np.mean(epoch_pixel)),
            "ssim": float(np.mean(epoch_ssim)),
            "psnr": float(np.mean(epoch_psnr)),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(logs_dir, "training_log.jsonl"), "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
              f"Average loss: {avg_loss:.6f}")

        # Scheduler step (if enabled)
        if scheduler is not None:
            scheduler.step()

        # Checkpointing
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

    # End of training: dump a summary JSON
    writer.flush()
    writer.close()
    log_data = {
        "training_time_minutes": (time.time() - start_wall) / 60.0,
        "config": config,
        "parameter_count": nca_param_count,
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
