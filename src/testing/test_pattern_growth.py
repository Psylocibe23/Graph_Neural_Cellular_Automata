import os
import torch
from utils.nca_init import make_seed
from modules.nca import NeuralCA
from modules.graph_augmentation import GraphAugmentation
from utils.image import load_single_target_image
from utils.config import load_config
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def save_img(img, path, upscale=4):
    """Save RGBA canvas, optionally upscaled, as PNG."""
    if hasattr(img, "detach"):
        img = img.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img = img[:4].permute(1,2,0).numpy().clip(0,1)
    if upscale > 1:
        import torch.nn.functional as F
        img_t = torch.tensor(img).permute(2,0,1).unsqueeze(0)
        img_t = F.interpolate(img_t, scale_factor=upscale, mode='bilinear', align_corners=False)[0]
        img = img_t.permute(1,2,0).numpy()
        img = np.clip(img,0,1)
    plt.imsave(path, img)

def save_heatmap(attn_map, path, upscale=4):
    """Save the attention heatmap as PNG."""
    img = attn_map.detach().cpu().numpy()
    if img.ndim == 3:
        img = img[0]
    if upscale > 1:
        img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        img_t = F.interpolate(img_t, scale_factor=upscale, mode='bilinear', align_corners=False)[0,0]
        img = img_t.numpy()
    plt.imsave(path, img, cmap="hot")

def main():
    # ---- SETTINGS ----
    config = load_config('configs/test_config.json')
    device = config['misc']['device']
    n_channels = config['model']['n_channels']
    img_size = config['data']['img_size']
    steps = 100  # Number of growth steps
    upscale = 4  # For better video frames
    model_type = 'graph_augmented'   # or 'classic_nca'
    checkpoint_path = "outputs/graph_augmented/gecko/checkpoints/nca_graphaug_epoch15.pt"
    save_dir = "outputs/graph_augmented/growth_video/gecko_graphaug"
    os.makedirs(save_dir, exist_ok=True)
    attn_save_dir = os.path.join(save_dir, "attention")
    os.makedirs(attn_save_dir, exist_ok=True)

    # ---- LOAD MODEL ----
    model = NeuralCA(
        n_channels=n_channels,
        update_hidden=config['model']['update_mlp']['hidden_dim'],
        update_layers=config['model']['update_mlp']['layers'],
        layer_norm=config['model']['layer_norm'],
        img_size=img_size,
        device=device
    ).to(device)

    use_graph_aug = model_type == 'graph_augmented'
    if use_graph_aug:
        graph_aug = GraphAugmentation(
            n_channels=n_channels,
            d_model=config.get("graph_augmentation", {}).get("d_model", 16),
            attention_radius=config.get("graph_augmentation", {}).get("attention_radius", 4),
            num_neighbors=config.get("graph_augmentation", {}).get("num_neighbors", 8),
            gating_hidden=config.get("graph_augmentation", {}).get("gating_hidden", 32)
        ).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        graph_aug.load_state_dict(ckpt['graph_aug_state'])
        model.eval()
        graph_aug.eval()
    else:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

    # ---- PREPARE SEED ----
    seed = make_seed(n_channels, img_size, batch_size=1, device=device)

    # ---- GROW AND SAVE FRAMES ----
    state = seed.clone()
    with torch.no_grad():
        for step in range(steps+1):
            img_path = os.path.join(save_dir, f"frame_{step:03d}.png")
            save_img(state[:, :4], img_path, upscale=upscale)
            # Attention map (for graph-augmented only)
            if use_graph_aug:
                _, attn_map = graph_aug(state, return_attention_map=True)
                attn_map_path = os.path.join(attn_save_dir, f"attn_{step:03d}.png")
                save_heatmap(attn_map, attn_map_path, upscale=upscale)
                graph_message = _
                state = model(state + graph_message, fire_rate=1.0)
            else:
                state = model(state, fire_rate=1.0)
    print(f"All frames saved in {save_dir}")
    if use_graph_aug:
        print(f"Attention maps saved in {attn_save_dir}")

if __name__ == "__main__":
    main()
