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
from modules.graph_augmentation import GraphAugmentation
from torch.utils.tensorboard import SummaryWriter
from utils.utility_functions import count_parameters, damage_batch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1,2,0).numpy().clip(0,1)
    plt.imsave(filename, img)

def main():
    start_time = time.time()
    config = load_config("configs/config.json")
    # Directories for regeneration experiments
    active_target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "graph_augmented", "regeneration", active_target_name)
    results_dir = os.path.join(base_dir, "images")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    config["logging"]["results_dir"] = results_dir
    config["logging"]["checkpoint_dir"] = ckpt_dir

    device = config["misc"]["device"]
    torch.manual_seed(config["misc"]["seed"])

    target = load_single_target_image(config).to(device)
    img_size = config["data"]["img_size"]
    n_channels = config["model"]["n_channels"]

    writer = SummaryWriter(log_dir=tb_dir)

    # Seed function for pool
    def seed_fn(batch_size=1):
        return make_seed(n_channels, img_size, batch_size, device)

    # NCA and Graph Augmentation modules
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config["model"]["update_mlp"]["hidden_dim"],
        update_layers=config["model"]["update_mlp"]["layers"],
        layer_norm=config["model"]["layer_norm"],
        img_size=img_size,
        device=device
    ).to(device)

    graph_aug = GraphAugmentation(
        n_channels=n_channels,
        d_model=config.get("graph_augmentation", {}).get("d_model", 16),
        attention_radius=config.get("graph_augmentation", {}).get("attention_radius", 4),
        num_neighbors=config.get("graph_augmentation", {}).get("num_neighbors", 8),
        gating_hidden=config.get("graph_augmentation", {}).get("gating_hidden", 32)
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(graph_aug.parameters()),
        lr=config["training"]["learning_rate"]
    )

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
    damage_patch_size = config.get("training", {}).get("damage_patch_size", 10)

    # Metrics recording
    epoch_losses = []
    pixel_scores = []
    ssim_scores = []
    psnr_scores = []

    pixel_scores_pre = []
    ssim_scores_pre = []
    psnr_scores_pre = []

    loss_pre_epoch = []
    loss_post_epoch = []
    regen_gap_epoch = []

    tol = 0.05

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")
    for epoch in trange(1, total_epochs + 1, desc="Epochs"):
        avg_loss = 0.0
        epoch_pixel_scores_pre = []
        epoch_ssim_scores_pre = []
        epoch_psnr_scores_pre = []

        epoch_pixel_scores = []
        epoch_ssim_scores = []
        epoch_psnr_scores = []

        loss_pre_steps = []
        loss_post_steps = []
        regen_gap_steps = []

        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            idx, batch = pool.sample(batch_size)
            batch = batch.to(device)
            nca_steps = torch.randint(nca_steps_min, nca_steps_max+1, (1,)).item()
            nca_steps1 = nca_steps // 2

            # 1. Pre-damage rollout
            for _ in range(nca_steps1):
                graph_message = graph_aug(batch)
                batch = model(batch + graph_message, fire_rate=fire_rate)

            # --- Metrics BEFORE DAMAGE ---
            pred_pre = batch[:, :4, :, :]
            target_expand = target.unsqueeze(0).expand_as(pred_pre)
            loss_pre = F.mse_loss(pred_pre, target_expand)
            pred_pre_img = pred_pre[0].detach().cpu().numpy()
            target_img = target.cpu().numpy()
            diff_pre = np.abs(pred_pre_img[:4] - target_img[:4])
            pixel_perfect_pre = (diff_pre < tol).all(axis=0).mean()
            pred_pre_np = pred_pre[0, :3].detach().cpu().permute(1,2,0).numpy().clip(0, 1)
            target_np = target[:3].cpu().permute(1,2,0).numpy().clip(0, 1)
            epoch_pixel_scores_pre.append(float(pixel_perfect_pre))
            epoch_ssim_scores_pre.append(ssim(pred_pre_np, target_np, data_range=1.0, channel_axis=-1))
            epoch_psnr_scores_pre.append(psnr(pred_pre_np, target_np, data_range=1.0))
            loss_pre_steps.append(loss_pre.item())

            # 2. Apply damage
            batch = damage_batch(batch, damage_size=damage_patch_size)

            # 3. Post-damage (regeneration) rollout
            nca_steps2 = nca_steps - nca_steps1
            for _ in range(nca_steps2):
                graph_message = graph_aug(batch)
                batch = model(batch + graph_message, fire_rate=fire_rate)

            # --- Metrics AFTER REGENERATION ---
            pred_post = batch[:, :4, :, :]
            loss_post = F.mse_loss(pred_post, target_expand)
            pred_post_img = pred_post[0].detach().cpu().numpy()
            diff_post = np.abs(pred_post_img[:4] - target_img[:4])
            pixel_perfect_post = (diff_post < tol).all(axis=0).mean()
            pred_post_np = pred_post[0, :3].detach().cpu().permute(1,2,0).numpy().clip(0, 1)
            epoch_pixel_scores.append(float(pixel_perfect_post))
            epoch_ssim_scores.append(ssim(pred_post_np, target_np, data_range=1.0, channel_axis=-1))
            epoch_psnr_scores.append(psnr(pred_post_np, target_np, data_range=1.0))
            loss_post_steps.append(loss_post.item())

            # Regeneration gap
            regen_gap = loss_post.item() - loss_pre.item()
            regen_gap_steps.append(regen_gap)
            avg_loss += loss_post.item()

            # Training (post-damage)
            optimizer.zero_grad()
            loss_post.backward()
            optimizer.step()
            pool.replace(idx, batch)

            # --- TensorBoard logging per step ---
            global_step = (epoch - 1) * steps_per_epoch + step
            writer.add_scalar('Loss/train', loss_post.item(), global_step)
            if (step+1) % visualize_interval == 0:
                img = pred_post[0, :3].detach().cpu().clamp(0,1)
                writer.add_image('Predicted/sample', img, global_step)
            if (step+1) % log_interval == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                      f"Loss: {loss_post.item():.5f}")
            if (step+1) % visualize_interval == 0:
                save_path = os.path.join(results_dir, f"epoch{epoch}_step{step+1}.png")
                save_grid_as_image(batch[0], save_path)
                print(f"Saved sample grid to {save_path}")
                save_comparison(target, pred_post[0], f"{epoch}_step{step+1}", results_dir, upscale=4)
                print(f"Saved comparison image at epoch {epoch}, step {step+1}")

        avg_loss /= steps_per_epoch
        epoch_losses.append(avg_loss)
        pixel_scores.append(float(np.mean(epoch_pixel_scores)))
        ssim_scores.append(float(np.mean(epoch_ssim_scores)))
        psnr_scores.append(float(np.mean(epoch_psnr_scores)))

        pixel_scores_pre.append(float(np.mean(epoch_pixel_scores_pre)))
        ssim_scores_pre.append(float(np.mean(epoch_ssim_scores_pre)))
        psnr_scores_pre.append(float(np.mean(epoch_psnr_scores_pre)))

        loss_pre_epoch.append(np.mean(loss_pre_steps))
        loss_post_epoch.append(np.mean(loss_post_steps))
        regen_gap_epoch.append(np.mean(regen_gap_steps))

        writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
              f"Average loss: {avg_loss:.6f}")

        # Save model checkpoint at interval
        ckpt_interval = config["logging"].get("checkpoint_interval_epochs", 5)
        if epoch % ckpt_interval == 0 or epoch == total_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"nca_graphaug_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "graph_aug_state": graph_aug.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"Model checkpoint saved to {ckpt_path}")

    writer.close()
    log_data = {
        "training_time_minutes": (time.time() - start_time) / 60,
        "config": config,
        "parameter_count": count_parameters(model) + (count_parameters(graph_aug) if "graph_aug" in locals() else 0),
        "graph_augmentation_parameters": count_parameters(graph_aug),
        "seed": config["misc"].get("seed"),

        # Losses
        "initial_loss_pre_damage": float(loss_pre_epoch[0]) if loss_pre_epoch else None,
        "final_loss_pre_damage": float(loss_pre_epoch[-1]) if loss_pre_epoch else None,
        "epoch_losses_pre_damage": loss_pre_epoch,
        "initial_loss_post_regen": float(loss_post_epoch[0]) if loss_post_epoch else None,
        "final_loss_post_regen": float(loss_post_epoch[-1]) if loss_post_epoch else None,
        "epoch_losses_post_regen": loss_post_epoch,

        # Regeneration gap (post - pre)
        "average_regeneration_gap": float(np.mean(regen_gap_epoch)) if regen_gap_epoch else None,
        "regeneration_gap_per_epoch": regen_gap_epoch,

        # Pixel Perfection
        "average_pixel_perfection_post_regen": float(np.mean(pixel_scores)) if pixel_scores else None,
        "pixel_perfection_per_epoch_post_regen": pixel_scores,

        # SSIM and PSNR post-regeneration
        "SSIM description": "Structural Similarity Index. Measures perceptual similarity (luminance, contrast, structure); higher is better, max=1",
        "average_ssim_post_regen": float(np.mean(ssim_scores)) if ssim_scores else None,
        "ssim_per_epoch_post_regen": ssim_scores,
        "PSNR description": "Peak Signal-to-Noise Ratio in dB. Higher means lower pixel error and better fidelity",
        "average_psnr_post_regen": float(np.mean(psnr_scores)) if psnr_scores else None,
        "psnr_per_epoch_post_regen": psnr_scores,

        "log_pre_damage": {
            "average_pixel_perfection_pre_damage": float(np.mean(pixel_scores_pre)) if pixel_scores_pre else None,
            "pixel_perfection_per_epoch_pre_damage": pixel_scores_pre,
            "average_ssim_pre_damage": float(np.mean(ssim_scores_pre)) if ssim_scores_pre else None,
            "ssim_per_epoch_pre_damage": ssim_scores_pre,
            "average_psnr_pre_damage": float(np.mean(psnr_scores_pre)) if psnr_scores_pre else None,
            "psnr_per_epoch_pre_damage": psnr_scores_pre,
        },
    }

    log_path = os.path.join(logs_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved training log to {log_path}")

if __name__ == "__main__":
    main()
