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
from modules.graph_augmentation import GraphAugmentation
from torch.utils.tensorboard import SummaryWriter



def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1,2,0).numpy().clip(0,1)
    plt.imsave(filename, img)

def main():
    config = load_config("configs/config.json")
    # Set up special folder
    active_target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "graph_augmented", active_target_name)
    results_dir = os.path.join(base_dir, "images")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs") 
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
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

    # Build NCA (classic) and graph augmentation module
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config["model"]["update_mlp"]["hidden_dim"],
        update_layers=config["model"]["update_mlp"]["layers"],
        layer_norm=config["model"]["layer_norm"],
        img_size=img_size,
        device=device
    ).to(device)
    
    # --- Graph augmentation hyperparameters ---
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")
    for epoch in trange(1, total_epochs + 1, desc="Epochs"):
        avg_loss = 0.0
        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            idx, batch = pool.sample(batch_size)
            batch = batch.to(device)
            nca_steps = torch.randint(nca_steps_min, nca_steps_max+1, (1,)).item()
            for _ in range(nca_steps):
                # Classic: batch = model(batch, fire_rate=fire_rate)
                # Augmented: add graph message before NCA update
                graph_message = graph_aug(batch)
                batch = model(batch + graph_message, fire_rate=fire_rate)  # add the gated mid-range message
            pred = batch[:, :4, :, :]
            target_expand = target.unsqueeze(0).expand_as(pred)
            loss = F.mse_loss(pred, target_expand)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pool.replace(idx, batch)
            avg_loss += loss.item()

            # --- TensorBoard logging per step ---
            global_step = (epoch - 1) * steps_per_epoch + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            # log sample prediction as image (every so often)
            if (step+1) % visualize_interval == 0:
                img = pred[0, :3].detach().cpu().clamp(0,1)  # RGB only
                writer.add_image('Predicted/sample', img, global_step)

            if (step+1) % log_interval == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                      f"Loss: {loss.item():.5f}")
            if (step+1) % visualize_interval == 0:
                save_path = os.path.join(config["logging"]["results_dir"],
                                         f"epoch{epoch}_step{step+1}.png")
                save_grid_as_image(batch[0], save_path)
                print(f"Saved sample grid to {save_path}")
                save_comparison(target, pred[0], f"{epoch}_step{step+1}", config["logging"]["results_dir"], upscale=4)
                print(f"Saved comparison image at epoch {epoch}, step {step+1}")
        avg_loss /= steps_per_epoch
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
              f"Average loss: {avg_loss:.6f}")

        ckpt_interval = config["logging"].get("checkpoint_interval_epochs", 5)
        if epoch % ckpt_interval == 0 or epoch == total_epochs:
            ckpt_path = os.path.join(config["logging"]["checkpoint_dir"],
                                    f"nca_graphaug_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "graph_aug_state": graph_aug.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"Model checkpoint saved to {ckpt_path}")

    writer.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed.")

if __name__ == "__main__":
    main()
