import os
import torch
import json
import numpy as np
import time
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import trange
from utils.config import load_config
from utils.image import load_single_target_image
from utils.nca_init import make_seed
from training.pool import SamplePool
from modules.nca import NeuralCA
from utils.visualize import save_comparison
from torch.utils.tensorboard import SummaryWriter
from utils.utility_functions import count_parameters
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1,2,0).numpy().clip(0,1)
    plt.imsave(filename, img)

def masked_loss(pred, target):
    # pred: [B, 4, H, W], target: [B, 4, H, W]
    alive_mask = (pred[:, 3:4, :, :] > 0.1).float()
    mse = ((pred - target) ** 2) * alive_mask
    # sum over channel, height, width; mean over batch
    per_sample_loss = mse.sum(dim=(1,2,3)) / (alive_mask.sum(dim=(1,2,3)) + 1e-8)  # [B]
    return per_sample_loss

def save_debug_canvas(state, path, idx=0):
    # Save RGB and alpha channel of sample idx from a batch state
    with torch.no_grad():
        img = state[idx, :3].detach().cpu().clamp(0,1)
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

def main():
    start_time = time.time()
    config = load_config("configs/config.json")
    active_target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "classic_nca", "train_inter_loss", active_target_name)
    results_dir = os.path.join(base_dir, "images")
    inter_dir = os.path.join(base_dir, "intermediate")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    config["logging"]["results_dir"] = results_dir
    config["logging"]["checkpoint_dir"] = ckpt_dir

    device = config["misc"]["device"]
    torch.manual_seed(config["misc"]["seed"])

    target = load_single_target_image(config).to(device)  # [4, H, W]
    img_size = config["data"]["img_size"]
    n_channels = config["model"]["n_channels"]

    writer = SummaryWriter(log_dir=tb_dir)

    def seed_fn(batch_size=1):
        return make_seed(n_channels, img_size, batch_size, device)

    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config["model"]["update_mlp"]["hidden_dim"],
        layer_norm=config["model"]["layer_norm"],
        img_size=img_size,
        device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    nca_param_count = count_parameters(model)

    pool = SamplePool(config["training"]["pool_size"], seed_fn, device=device)

    total_epochs = config["training"]["num_epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]
    batch_size = config["training"]["batch_size"]
    nca_steps_min = config["training"]["nca_steps_min"]
    nca_steps_max = config["training"]["nca_steps_max"]
    fire_rate = config["model"]["fire_rate"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    visualize_interval = config["logging"]["visualize_interval"]

    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    epoch_losses = []
    pixel_scores = []
    ssim_scores = []
    psnr_scores = []

    reset_prob = 1.0  # Increase for stricter training
    tol = 0.05

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")
    for epoch in trange(1, total_epochs + 1, desc="Epochs"):
        avg_loss = 0.0
        epoch_pixel_scores = []
        epoch_ssim_scores = []
        epoch_psnr_scores = []

        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            idx, batch = pool.sample(batch_size)
            batch = batch.to(device)

            # NCA rollout with random steps per sample
            nca_steps = torch.randint(nca_steps_min, nca_steps_max+1, (batch_size,), device=device)
            state = batch.clone()
            max_steps = nca_steps.max().item()
            for t in range(1, max_steps+1):
                state = model(state, fire_rate=fire_rate)

            # Target shape: [4, H, W]; Expand to [B, 4, H, W]
            target_expand = target.unsqueeze(0).expand_as(state[:, :4])

            # Masked per-sample loss
            per_sample_loss = masked_loss(state[:, :4], target_expand)  # shape [B]
            loss = per_sample_loss.mean()

            # Reset *the sample with highest loss* in the batch to a fresh seed
            max_loss_idx = torch.argmax(per_sample_loss).item()
            state[max_loss_idx] = seed_fn(1)[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pool.replace(idx, state)

            # --- Logging and visualization (unchanged) ---
            pred = state[:, :4]
            pred_img = pred[0].detach().cpu().numpy()
            target_img = target.cpu().numpy()
            tol = 0.05
            diff = np.abs(pred_img[:4] - target_img[:4])
            pixel_perfect = (diff < tol).all(axis=0).mean()
            epoch_pixel_scores.append(float(pixel_perfect))
            pred_np = pred[0, :3].detach().cpu().permute(1,2,0).numpy().clip(0, 1)
            target_np = target[:3].cpu().permute(1,2,0).numpy().clip(0, 1)
            epoch_ssim_scores.append(ssim(pred_np, target_np, data_range=1.0, channel_axis=-1))
            epoch_psnr_scores.append(psnr(pred_np, target_np, data_range=1.0))
            avg_loss += loss.item()

            global_step = (epoch - 1) * steps_per_epoch + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            if (step+1) % visualize_interval == 0:
                img = pred[0, :3].detach().cpu().clamp(0,1)
                writer.add_image('Predicted/sample', img, global_step)
                # Optionally save alpha debug etc.

            if (step+1) % log_interval == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                      f"Loss: {loss.item():.5f}")

            # (Optional) Save grid images for debugging
            if (step+1) % visualize_interval == 0:
                save_path = os.path.join(results_dir, f"epoch{epoch}_step{step+1}.png")
                save_grid_as_image(state[0], save_path)
                save_comparison(target, pred[0], f"{epoch}_step{step+1}", results_dir, upscale=4)

        avg_loss /= steps_per_epoch
        epoch_losses.append(avg_loss)
        pixel_scores.append(float(np.mean(epoch_pixel_scores)))
        ssim_scores.append(float(np.mean(epoch_ssim_scores)))
        psnr_scores.append(float(np.mean(epoch_psnr_scores)))
        writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
              f"Average loss: {avg_loss:.6f}")

        # Save checkpoint
        ckpt_interval = config["logging"].get("checkpoint_interval_epochs", 5)
        if epoch % ckpt_interval == 0 or epoch == total_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"nca_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"Model checkpoint saved to {ckpt_path}")


    writer.close()
    log_data = {
        "training_time_minutes": (time.time() - start_time)/60,
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
    }

    log_path = os.path.join(logs_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved training log to {log_path}")

if __name__ == "__main__":
    main()
