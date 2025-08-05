import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.nca_init import make_seed
from modules.nca import NeuralCA
from utils.config import load_config
from utils.image import load_single_target_image
import torch.nn.functional as F


def save_img(img, path, upscale=4):
    # Save RGBA image, upscaled, masking out dead cells (alpha < 0.1)
    if hasattr(img, "detach"):
        img = img.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img_np = img[:4].permute(1, 2, 0).numpy().copy()  # [H,W,4]
    alpha = img_np[..., 3:4]
    mask = (alpha > 0.1).astype(np.float32)
    img_np[..., :3] *= mask
    img_np[..., 3:] = mask

    # Upscale AFTER masking, using torch
    if upscale > 1:
        img_t = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)  # [1,4,H,W]
        img_t = F.interpolate(img_t, scale_factor=upscale, mode='bilinear', align_corners=False)[0]
        img_np = img_t.permute(1,2,0).numpy()
        img_np = np.clip(img_np, 0, 1)

    assert img_np.ndim == 3 and img_np.shape[2] in (3,4), f"Got shape {img_np.shape}"
    plt.imsave(path, img_np)


def main():
    # --- SETTINGS ---
    config = load_config('configs/config.json')
    device = config["misc"]["device"]
    n_channels = config["model"]["n_channels"]
    img_size = config["data"]["img_size"]
    steps = 50
    upscale = 4
    target_name = os.path.splitext(config["data"]["active_target"])[0]
    ckpt_path = f"outputs/classic_nca/train_inter_loss/{target_name}/checkpoints/nca_epoch5.pt"
    save_dir = f"outputs/classic_nca/test_growth/{target_name}"
    os.makedirs(save_dir, exist_ok=True)

    # --- LOAD MODEL ---
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config["model"]["update_mlp"]["hidden_dim"],
        layer_norm=config["model"]["layer_norm"],
        img_size=img_size,
        device=device
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- PREPARE SEED ---
    seed = make_seed(n_channels, img_size, batch_size=1, device=device)

    # --- GROWTH ROLLOUT ---
    state = seed.clone()
    all_imgs = []
    with torch.no_grad():
        for t in range(steps+1):
            frame_path = os.path.join(save_dir, f"frame_{t:03d}.png")
            save_img(state[:, :4], frame_path, upscale=upscale)
            all_imgs.append(state[:, :4][0].cpu())
            state = model(state, fire_rate=1.0)   # fire_rate=1.0 ensures all cells are updated

    print(f"All growth frames saved to {save_dir}")

    # --- DISPLAY A GRID OF SELECTED FRAMES ---
    select_steps = [1,2,4,8,16,32,49]  # You can change these as you want
    n = len(select_steps)
    plt.figure(figsize=(n*2,2.5))
    for i, step in enumerate(select_steps):
        plt.subplot(1,n,i+1)
        img = all_imgs[step][:3].permute(1,2,0).numpy().clip(0,1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Step {step}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
