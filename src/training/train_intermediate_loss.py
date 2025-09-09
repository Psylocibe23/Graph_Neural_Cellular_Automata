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
from utils.damage import apply_damage_policy_ 

"""
Classic Neural CA trainer (no graph), with damage curriculum and
premultiplied-RGBA loss (Distill-style).

- First run: can resume from an exact epoch of the OLD run
  (outputs/classic_nca/train_inter_loss/<target>/checkpoints/nca_epoch{E}.pt)
  using config.training.resume_epoch_exact or training.resume_ckpt_path.
- Subsequent runs: auto-resume from the latest checkpoint in the NEW folder:
  outputs/classic_nca/train_inter_loss_damage/<target>/checkpoints/
"""

# -------------------------- Loss (premultiplied RGBA) --------------------------
def loss_premult_rgba(pred_rgba, target_rgba):
    """
    pred_rgba: [B,4,H,W] raw model RGBA (RGB NOT premultiplied yet)
    target_rgba: [B,4,H,W] premultiplied RGBA target
    Returns: per-sample vector [B]
    """
    pred_rgb_prem = pred_rgba[:, :3] * pred_rgba[:, 3:4]
    pred_pm = torch.cat([pred_rgb_prem, pred_rgba[:, 3:4]], dim=1)
    return F.mse_loss(pred_pm, target_rgba, reduction="none").mean(dim=(1, 2, 3))


# --------------------------------- Utilities -----------------------------------
def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1, 2, 0).numpy().clip(0, 1)
    plt.imsave(filename, img)


def extract_epoch_num(ckpt_filename):
    m = re.search(r'epoch(\d+)', ckpt_filename)
    return int(m.group(1)) if m else -1


# ----------------------------------- Main --------------------------------------
def main():
    start_wall = time.time()
    config = load_config("configs/config.json")

    target_name = os.path.splitext(config["data"]["active_target"])[0]

    # NEW damage run folder
    base_dir = os.path.join("outputs", "classic_nca", "train_inter_loss_damage", target_name)
    results_dir = os.path.join(base_dir, "images")
    inter_dir = os.path.join(base_dir, "intermediate")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs")
    logs_dir = os.path.join(base_dir, "logs")
    for d in [results_dir, inter_dir, ckpt_dir, tb_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    config["logging"]["results_dir"] = results_dir
    config["logging"]["checkpoint_dir"] = ckpt_dir

    device = config["misc"]["device"]
    torch.manual_seed(config["misc"]["seed"])

    # Damage config (recommend: start_epoch=100)
    damage_cfg = config.get("damage", {})
    if "start_epoch" in damage_cfg:
        print(f"[damage] start_epoch={damage_cfg['start_epoch']}")
    else:
        print("[damage] No start_epoch in config.damage (default behavior inside policy).")

    # Data / target (premultiply RGB by alpha once)
    target = load_single_target_image(config).to(device)  # [4,H,W] in [0,1]
    target[:3] = target[:3] * target[3:4]

    img_size = int(config["data"]["img_size"])
    n_channels = int(config["model"]["n_channels"])

    # TensorBoard
    writer = SummaryWriter(log_dir=tb_dir)

    # Seed: alpha=1 at center; small hidden noise
    def seed_fn(batch_size=1):
        g = torch.zeros(batch_size, n_channels, img_size, img_size, device=device)
        cy = img_size // 2; cx = img_size // 2
        g[:, 3:4, cy, cx] = 1.0
        if n_channels > 4:
            g[:, 4:, cy, cx] = 0.01 * torch.randn_like(g[:, 4:, cy, cx])
        return g

    # Model (classic NCA)
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=int(config["model"]["update_mlp"]["hidden_dim"]),
        img_size=img_size,
        update_gain=float(config["model"].get("update_gain", 0.1)),
        alpha_thr=float(config["model"].get("alpha_thr", 0.1)),
        use_groupnorm=bool(config["model"].get("use_groupnorm", True)),
        device=device
    ).to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    # Scheduler (optional)
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

    # Pool
    pool = SamplePool(int(config["training"]["pool_size"]), seed_fn, device=device)

    # Training params
    total_epochs = int(config["training"]["num_epochs"])
    steps_per_epoch = int(config["training"]["steps_per_epoch"])
    batch_size = int(config["training"]["batch_size"])
    short_min = int(config["training"]["nca_steps_min"])
    short_max = int(config["training"]["nca_steps_max"])
    long_prob = float(config["training"].get("long_rollout_prob", 0.25))
    long_min = int(config["training"].get("long_rollout_steps_min", 200))
    long_max = int(config["training"].get("long_rollout_steps_max", 400))
    log_interval = int(config["logging"]["log_interval"])
    visualize_interval = int(config["logging"]["visualize_interval"])
    ckpt_interval = int(config["logging"].get("checkpoint_interval_epochs", 25))

    fr_min = float(config["training"].get("fire_rate_min", 0.5))
    fr_max = float(config["training"].get("fire_rate_max", 1.0))

    reset_worst_prob = float(config["training"].get("reset_worst_prob", 0.10))
    random_reseed_prob = float(config["training"].get("random_reseed_prob", 0.05))

    # ----------------------- Resume logic (exact epoch support) -----------------------
    # Subsequent runs: resume from NEW folder's latest if present.
    new_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "nca_epoch*.pt")), key=extract_epoch_num)
    resume_epoch_exact = config["training"].get("resume_epoch_exact")  # e.g., 100 or 200
    resume_ckpt_path = config["training"].get("resume_ckpt_path", "")  # explicit path override

    if new_ckpts:
        last_ckpt = new_ckpts[-1]
        print(f"[resume] Found checkpoints in new folder. Using latest: {last_ckpt}")
    else:
        # First run of the damage trainer: take an exact epoch (or explicit path) from OLD folder
        old_dir = os.path.join("outputs", "classic_nca", "train_inter_loss", target_name, "checkpoints")
        if resume_ckpt_path:
            last_ckpt = resume_ckpt_path
        elif resume_epoch_exact is not None:
            last_ckpt = os.path.join(old_dir, f"nca_epoch{int(resume_epoch_exact)}.pt")
        else:
            # Fallback: use latest from OLD folder if no exact epoch specified
            old_ckpts = sorted(glob.glob(os.path.join(old_dir, "nca_epoch*.pt")), key=extract_epoch_num)
            if not old_ckpts:
                raise FileNotFoundError(
                    f"No checkpoints found in {ckpt_dir} or {old_dir}. "
                    f"Set training.resume_epoch_exact or training.resume_ckpt_path."
                )
            last_ckpt = old_ckpts[-1]
        print(f"[resume] No new-folder checkpoints. Loading from: {last_ckpt}")

    # Load checkpoint
    checkpoint = torch.load(last_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:   print(f"[resume] missing keys (initialized fresh): {missing}")
    if unexpected: print(f"[resume] unexpected keys (ignored): {unexpected}")

    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except Exception as e:
        print(f"[warn] optimizer state not compatible, reinitializing: {e}")

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception as e:
            print(f"[warn] scheduler state not compatible, reinitializing: {e}")

    # Continue epoch count from the loaded checkpoint
    start_epoch = int(checkpoint["epoch"]) + 1
    print(f"Resuming from epoch {checkpoint['epoch']} -> start at epoch {start_epoch}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6g}")

    # Param count
    nca_param_count = count_parameters(model)
    print(f"Params: {nca_param_count}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training until {total_epochs} epochs")

    # ------------------------------- Training loop --------------------------------
    epoch_losses, pixel_scores, ssim_scores, psnr_scores = [], [], [], []

    for epoch in trange(start_epoch, total_epochs + 1, desc="Epochs"):
        avg_loss = 0.0
        epoch_pixel, epoch_ssim, epoch_psnr = [], [], []

        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            # Sample batch from pool
            idx, batch = pool.sample(batch_size)
            state = batch.to(device)

            # Apply damage BEFORE rollout (policy checks damage_cfg['start_epoch'])
            apply_damage_policy_(state, damage_cfg, epoch)

            # Rollout schedule
            if random.random() < long_prob:
                steps_lo, steps_hi = long_min, long_max
            else:
                steps_lo, steps_hi = short_min, short_max

            nca_steps = torch.randint(steps_lo, steps_hi + 1, (batch_size,), device=device)
            max_steps = int(nca_steps.max().item())

            # Rollout with randomized fire-rate per step
            for t in range(max_steps):
                mask = (nca_steps > t)
                if mask.any():
                    fire_rate = float(torch.empty(1, device=device).uniform_(fr_min, fr_max).item())
                    state[mask] = model(state[mask], fire_rate=fire_rate)

            # Premultiplied loss
            target_expand = target.unsqueeze(0).expand_as(state[:, :4])
            per_sample = loss_premult_rgba(state[:, :4], target_expand)
            loss = per_sample.mean()

            # Optional stability on premultiplied RGBA
            if bool(config["training"].get("stability_enabled", True)):
                stab_thresh = float(config["training"].get("stability_threshold", 0.01))
                with torch.no_grad():
                    close_mask = (per_sample < stab_thresh)
                if close_mask.any():
                    K = int(config["training"].get("stability_K", 24))
                    state_stab = state.clone()
                    for _ in range(K):
                        fr = float(torch.empty(1, device=device).uniform_(fr_min, fr_max).item())
                        state_stab[close_mask] = model(state_stab[close_mask], fire_rate=fr)
                    pred_stab_rgb_prem = state_stab[close_mask, :3] * state_stab[close_mask, 3:4]
                    pred_stab_rgba = torch.cat([pred_stab_rgb_prem, state_stab[close_mask, 3:4]], dim=1)
                    stab = F.mse_loss(pred_stab_rgba, target_expand[close_mask])
                    loss = loss + float(config["training"].get("stability_weight", 0.5)) * stab

            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Pool maintenance
            n_reset = int(reset_worst_prob * batch_size)
            worst_indices = torch.topk(per_sample, n_reset).indices if n_reset > 0 else None
            do_random_reseed = (random.random() < random_reseed_prob)
            rand_idx = torch.randint(0, batch_size, (1,), device=device).item() if do_random_reseed else None

            state_for_pool = state.detach()
            if (worst_indices is not None) or do_random_reseed:
                state_for_pool = state_for_pool.clone()
                if worst_indices is not None:
                    state_for_pool[worst_indices] = seed_fn(len(worst_indices)).detach()
                if do_random_reseed:
                    state_for_pool[rand_idx:rand_idx+1] = seed_fn(1).detach()
            pool.replace(idx, state_for_pool)

            # Metrics on premultiplied tensors
            pred = state[:, :4]
            pred_rgba = torch.cat([pred[:, :3] * pred[:, 3:4], pred[:, 3:4]], dim=1)
            tgt_rgba = target_expand

            p0 = pred_rgba[0].detach().cpu().numpy()
            t0 = tgt_rgba[0].detach().cpu().numpy()
            diff = np.abs(p0 - t0)
            pixel_perfect = (diff < 0.05).all(axis=0).mean()
            epoch_pixel.append(float(pixel_perfect))

            pred_np_rgb_prem = (p0[:3]).transpose(1, 2, 0).clip(0, 1)
            tgt_np_rgb_prem = (t0[:3]).transpose(1, 2, 0).clip(0, 1)
            epoch_ssim.append(ssim(pred_np_rgb_prem, tgt_np_rgb_prem, data_range=1.0, channel_axis=-1))
            epoch_psnr.append(psnr(pred_np_rgb_prem, tgt_np_rgb_prem, data_range=1.0))

            avg_loss += loss.item()

            # Logging
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

        # End of epoch
        avg_loss /= steps_per_epoch
        epoch_losses.append(avg_loss)
        pixel_scores.append(float(np.mean(epoch_pixel)))
        ssim_scores.append(float(np.mean(epoch_ssim)))
        psnr_scores.append(float(np.mean(epoch_psnr)))

        # per-epoch row
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

        # Checkpoint in NEW folder
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

    # Summary JSON
    writer.flush(); writer.close()
    with open(os.path.join(logs_dir, "training_log.json"), "w") as f:
        json.dump({
            "training_time_minutes": (time.time() - start_wall) / 60.0,
            "config": config,
            "parameter_count": nca_param_count,
            "seed": config["misc"].get("seed"),
            "initial_loss": float(epoch_losses[0]) if epoch_losses else None,
            "final_loss": float(epoch_losses[-1]) if epoch_losses else None,
            "epoch_losses": epoch_losses,
            "average_pixel_perfection": float(np.mean(pixel_scores)) if pixel_scores else None,
            "pixel_perfection_per_epoch": pixel_scores,
            "average_ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
            "ssim_per_epoch": ssim_scores,
            "average_psnr": float(np.mean(psnr_scores)) if psnr_scores else None,
            "psnr_per_epoch": psnr_scores
        }, f, indent=2)
    print(f"Saved training log to {os.path.join(logs_dir, 'training_log.json')}")


if __name__ == "__main__":
    main()
