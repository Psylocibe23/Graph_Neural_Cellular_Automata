import os
import torch
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



def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1,2,0).numpy().clip(0,1)
    plt.imsave(filename, img)
    
def main():
    # 1. Load config and set up per-target output dirs
    config = load_config("configs/config.json")
    active_target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "classic_nca", active_target_name)
    results_dir = os.path.join(base_dir, "images")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    config["logging"]["results_dir"] = results_dir
    config["logging"]["checkpoint_dir"] = ckpt_dir

    # 2. Set device, reproducibility
    device = config["misc"]["device"]
    torch.manual_seed(config["misc"]["seed"])

    # 3. Load target image
    target = load_single_target_image(config).to(device)  # [4, H, W]
    img_size = config["data"]["img_size"]
    n_channels = config["model"]["n_channels"]

    # 4. TensorBoard writer
    writer = SummaryWriter(log_dir=tb_dir)

    # 5. Seed function for pool
    def seed_fn(batch_size=1):
        return make_seed(n_channels, img_size, batch_size, device)

    # 6. Build NCA and optimizer
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config["model"]["update_mlp"]["hidden_dim"],
        update_layers=config["model"]["update_mlp"]["layers"],
        layer_norm=config["model"]["layer_norm"],
        img_size=img_size,
        device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    nca_param_count = count_parameters(model)
    
    # 7. Initialize pool
    pool = SamplePool(config["training"]["pool_size"], seed_fn, device=device)
    
    # 8. Training loop parameters
    total_epochs = config["training"]["num_epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]
    batch_size = config["training"]["batch_size"]
    nca_steps_min = config["training"]["nca_steps_min"]
    nca_steps_max = config["training"]["nca_steps_max"]
    fire_rate = config["model"]["fire_rate"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    visualize_interval = config["logging"]["visualize_interval"]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")
    for epoch in trange(1, total_epochs + 1, desc="Epochs"):
        avg_loss = 0.0
        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            idx, batch = pool.sample(batch_size)
            batch = batch.to(device)
            # Rollout for a random number of steps
            nca_steps = torch.randint(nca_steps_min, nca_steps_max+1, (1,)).item()
            for _ in range(nca_steps):
                batch = model(batch, fire_rate=fire_rate)
            # Loss: MSE between first 4 channels (RGBA) and target
            pred = batch[:, :4, :, :]
            target_expand = target.unsqueeze(0).expand_as(pred)
            loss = F.mse_loss(pred, target_expand)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update pool
            pool.replace(idx, batch)
            avg_loss += loss.item()

            # --- TensorBoard logging per step ---
            global_step = (epoch - 1) * steps_per_epoch + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            # log sample prediction as image
            if (step+1) % visualize_interval == 0:
                img = pred[0, :3].detach().cpu().clamp(0,1)  # RGB only
                writer.add_image('Predicted/sample', img, global_step)

            # Print progress
            if (step+1) % log_interval == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                      f"Loss: {loss.item():.5f}")
            # Save sample grid image
            if (step+1) % visualize_interval == 0:
                save_path = os.path.join(results_dir, f"epoch{epoch}_step{step+1}.png")
                save_grid_as_image(batch[0], save_path)
                print(f"Saved sample grid to {save_path}")
                save_comparison(target, pred[0], f"{epoch}_step{step+1}", results_dir, upscale=4)
                print(f"Saved comparison image at epoch {epoch}, step {step+1}")
        avg_loss /= steps_per_epoch
        writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
              f"Average loss: {avg_loss:.6f}")
        
        # Save model checkpoint at interval
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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed.")
    print(f"Classic NCA parameters: {nca_param_count:,}")

if __name__ == "__main__":
    main()
