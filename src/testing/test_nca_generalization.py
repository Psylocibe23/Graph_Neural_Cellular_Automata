import os
import torch
from utils.nca_init import make_seed
from modules.nca import NeuralCA
from utils.image import load_single_target_image
from utils.config import load_config
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import json
from utils.visualize import save_comparison


# 1. Load config and prepare device
config = load_config('configs/test_config.json')
device = config['misc']['device']
os.makedirs('outputs/tests/eye', exist_ok=True)

# 2. Build model and load weights
model = NeuralCA(
    n_channels=config["model"]["n_channels"],
    update_hidden=config["model"]["update_mlp"]["hidden_dim"],
    update_layers=config["model"]["update_mlp"]["layers"],
    layer_norm=config["model"]["layer_norm"],
    img_size=config["data"]["img_size"],
    device=device
).to(device)

ckpt = torch.load("outputs/classic_nca/squid/checkpoints/nca_epoch15.pt", map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# 3. Load target test image (not used during training)
target = load_single_target_image(config).to(device)  # expects config, not path!

# 4. Create a seed and rollout
img_size = config["data"]["img_size"]
n_channels = config["model"]["n_channels"]
seed = make_seed(n_channels, img_size, batch_size=1, device=device)
steps = 100
state = seed.clone()
with torch.no_grad():
    for i in range(steps):
        state = model(state, fire_rate=1.0)

# 5. Compute metrics
pred = state[:, :4, :, :]  # RGBA
target_exp = target.unsqueeze(0).expand_as(pred)
mse = F.mse_loss(pred, target_exp).item()
pred_np = pred[0, :3].detach().cpu().permute(1,2,0).numpy().clip(0, 1)
target_np = target[:3].cpu().permute(1,2,0).numpy().clip(0, 1)
ssim_score = ssim(pred_np, target_np, data_range=1.0, channel_axis=-1)
psnr_score = psnr(pred_np, target_np, data_range=1.0)

# 6. Save image comparison
results_dir = "outputs/tests/eye"
save_comparison(
    target=target, 
    pred=pred[0], 
    epoch="test",            
    out_dir=results_dir,
    filename="comparison.png",
    upscale=4
)

# 7. Save metrics
metrics = {
    "mse": mse,
    "ssim": float(ssim_score),
    "psnr": float(psnr_score),
    "steps": steps,
    "checkpoint": "outputs/classic_nca/squid/checkpoints/nca_epoch15.pt",
    "test_target": config["data"]["active_target"]
}
with open("outputs/tests/eye/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done! Metrics saved to outputs/tests/eye/metrics.json")
