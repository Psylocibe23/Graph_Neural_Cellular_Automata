import os
import torch
from utils.nca_init import make_seed
from modules.nca import NeuralCA
from modules.graph_augmentation import GraphAugmentation
from utils.image import load_single_target_image
from utils.config import load_config
from utils.visualize import save_comparison
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np


# ---- Load config and checkpoint ----
config = load_config('configs/test_config.json')
device = config['misc']['device']
n_channels = config['model']['n_channels']
img_size = config['data']['img_size']
test_target = config['data']['active_target']

# Build model
model = NeuralCA(
    n_channels=n_channels,
    update_hidden=config['model']['update_mlp']['hidden_dim'],
    update_layers=config['model']['update_mlp']['layers'],
    layer_norm=config['model']['layer_norm'],
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

# Load checkpoint (adjust path as needed)
ckpt_path = "outputs/graph_augmented/squid/checkpoints/nca_graphaug_epoch15.pt" 
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
graph_aug.load_state_dict(ckpt['graph_aug_state'])
model.eval()
graph_aug.eval()

# ---- Load the test image ----
target = load_single_target_image(config).to(device) 

# ---- Make seed and run rollout ----
seed = make_seed(n_channels, img_size, batch_size=1, device=device)
steps = 100
state = seed.clone()
with torch.no_grad():
    for i in range(steps):
        graph_message = graph_aug(state)
        state = model(state + graph_message, fire_rate=1.0) 

# ---- Compute metrics ----
pred = state[:, :4, :, :]
target_exp = target.unsqueeze(0).expand_as(pred)

mse = torch.nn.functional.mse_loss(pred, target_exp).item()
pred_np = pred[0, :3].detach().cpu().permute(1,2,0).numpy().clip(0, 1)
target_np = target[:3].cpu().permute(1,2,0).numpy().clip(0, 1)
ssim_val = ssim(pred_np, target_np, data_range=1.0, channel_axis=-1)
psnr_val = psnr(pred_np, target_np, data_range=1.0)

# ---- Save results ----
results_dir = "outputs/tests/eye_graph_augmented"
os.makedirs(results_dir, exist_ok=True)
save_comparison(target, pred[0], epoch="test", out_dir=results_dir, filename="comparison.png", upscale=4)

# Save metrics
metrics = {
    "mse": mse,
    "ssim": float(ssim_val),
    "psnr": float(psnr_val)
}
with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    import json
    json.dump(metrics, f, indent=2)

print(f"Saved comparison and metrics in {results_dir}")
