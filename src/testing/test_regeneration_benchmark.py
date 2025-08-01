import os
import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.image import load_single_target_image
from utils.config import load_config
from utils.nca_init import make_seed
from modules.nca import NeuralCA
from modules.graph_augmentation import GraphAugmentation
import matplotlib.pyplot as plt
import json

# ------------------- Damage Functions (to RGBA only) -------------------
def damage_square(rgba, size):
    img = rgba.clone()
    C, H, W = img.shape[-3:]
    top = np.random.randint(0, H-size)
    left = np.random.randint(0, W-size)
    img[..., top:top+size, left:left+size] = 0.0
    return img

def damage_line(rgba, orientation="horizontal", thickness=2):
    img = rgba.clone()
    C, H, W = img.shape[-3:]
    if orientation == "horizontal":
        row = np.random.randint(thickness, H-thickness)
        img[..., row:row+thickness, :] = 0.0
    else:
        col = np.random.randint(thickness, W-thickness)
        img[..., :, col:col+thickness] = 0.0
    return img

def damage_half(rgba, orientation="vertical"):
    img = rgba.clone()
    C, H, W = img.shape[-3:]
    if orientation == "vertical":
        img[..., :, :W//2] = 0.0
    else:
        img[..., :H//2, :] = 0.0
    return img

def apply_damage_to_state(state, damage_type):
    # state: [1, n_channels, H, W]
    damaged = state.clone()
    rgba = damaged[0, :4, :, :]  # [4, H, W]
    if damage_type == "small_square":
        rgba = damage_square(rgba, size=6)
    elif damage_type == "large_square":
        rgba = damage_square(rgba, size=20)
    elif damage_type == "horizontal_line":
        rgba = damage_line(rgba, "horizontal", thickness=2)
    elif damage_type == "vertical_line":
        rgba = damage_line(rgba, "vertical", thickness=2)
    elif damage_type == "half_vertical":
        rgba = damage_half(rgba, "vertical")
    elif damage_type == "half_horizontal":
        rgba = damage_half(rgba, "horizontal")
    else:
        raise ValueError(f"Unknown damage type: {damage_type}")
    damaged[0, :4, :, :] = rgba
    return damaged

# ------------------- Metrics -------------------
def get_metrics(pred, target, tol=0.05):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    diff = np.abs(pred_np[:4] - target_np[:4])
    pixel_perfect = (diff < tol).all(axis=0).mean()
    pred_rgb = pred_np[:3].transpose(1,2,0).clip(0,1)
    target_rgb = target_np[:3].transpose(1,2,0).clip(0,1)
    return {
        "mse": float(F.mse_loss(pred, target).item()),
        "ssim": float(ssim(pred_rgb, target_rgb, data_range=1.0, channel_axis=-1)),
        "psnr": float(psnr(pred_rgb, target_rgb, data_range=1.0)),
        "pixel_perfection": float(pixel_perfect)
    }

# ------------------- Visualization: Side-by-Side Damaged/Regenerated -------------------
def save_side_by_side(damaged, regenerated, damage_name, model_type, out_dir, upscale=4):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os

    # Ensure CPU and numpy
    if hasattr(damaged, "detach"): damaged = damaged.detach().cpu()
    if hasattr(regenerated, "detach"): regenerated = regenerated.detach().cpu()
    if damaged.ndim == 4: damaged = damaged[0]
    if regenerated.ndim == 4: regenerated = regenerated[0]
    # Upscale
    def upscale_img(img, scale):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3: img = img.unsqueeze(0)
            img = torch.nn.functional.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)[0]
            img = img.permute(1,2,0).numpy()
        else:
            img = np.clip(img,0,1)
            from PIL import Image
            img = (img*255).astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((img.size[0]*scale, img.size[1]*scale), Image.BILINEAR)
            img = np.asarray(img).astype(np.float32)/255.0
        return img
    d_img = upscale_img(damaged[:4], upscale)
    r_img = upscale_img(regenerated[:4], upscale)
    # Plot
    fig, axs = plt.subplots(1,2,figsize=(6,3))
    axs[0].imshow(d_img); axs[0].set_title('Damaged'); axs[0].axis("off")
    axs[1].imshow(r_img); axs[1].set_title('Regenerated'); axs[1].axis("off")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{model_type}_{damage_name}_regeneration.png"
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)

# ------------------- Test Script -------------------
def rollout_regen(model, initial, fire_rate, steps, target, graph_aug=None):
    # initial: [1, n_channels, H, W]
    x = initial.clone()
    metrics = []
    for t in range(steps):
        if graph_aug is not None:
            x = model(x + graph_aug(x), fire_rate=fire_rate)
        else:
            x = model(x, fire_rate=fire_rate)
        pred = x[:, :4, :, :][0]  # [4, H, W]
        m = get_metrics(pred, target)   
        metrics.append(m)
    return x, metrics

def main():
    # ---- Settings ----
    test_config = load_config('configs/test_config.json')
    target_name = os.path.splitext(test_config["data"]["active_target"])[0]
    out_dir = f"outputs/tests/regen_benchmark/{target_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load Target Image ----
    device = test_config["misc"]["device"]
    target = load_single_target_image(test_config).to(device)
    img_size = test_config["data"]["img_size"]
    n_channels = test_config["model"]["n_channels"]

    # ---- Load Models ----
    # Classic NCA
    model_ckpt = torch.load(
        f"outputs/classic_nca/regeneration/{target_name}/checkpoints/nca_epoch15.pt",
        map_location=device
    )
    classic_nca = NeuralCA(
        n_channels=n_channels,
        update_hidden=test_config["model"]["update_mlp"]["hidden_dim"],
        update_layers=test_config["model"]["update_mlp"]["layers"],
        layer_norm=test_config["model"]["layer_norm"],
        img_size=img_size,
        device=device,
    ).to(device)
    classic_nca.load_state_dict(model_ckpt["model_state"])
    classic_nca.eval()

    # Graph-augmented NCA
    graph_ckpt = torch.load(
        f"outputs/graph_augmented/regeneration/{target_name}/checkpoints/nca_graphaug_epoch15.pt",
        map_location=device
    )
    graph_nca = NeuralCA(
        n_channels=n_channels,
        update_hidden=test_config["model"]["update_mlp"]["hidden_dim"],
        update_layers=test_config["model"]["update_mlp"]["layers"],
        layer_norm=test_config["model"]["layer_norm"],
        img_size=img_size,
        device=device,
    ).to(device)
    graph_nca.load_state_dict(graph_ckpt["model_state"])
    graph_nca.eval()
    graph_aug = GraphAugmentation(
        n_channels=n_channels,
        d_model=test_config.get("graph_augmentation", {}).get("d_model", 16),
        attention_radius=test_config.get("graph_augmentation", {}).get("attention_radius", 4),
        num_neighbors=test_config.get("graph_augmentation", {}).get("num_neighbors", 8),
        gating_hidden=test_config.get("graph_augmentation", {}).get("gating_hidden", 32)
    ).to(device)
    graph_aug.load_state_dict(graph_ckpt["graph_aug_state"])
    graph_aug.eval()

    # ---- Damage Types ----
    damage_types = [
        "small_square", "large_square", "horizontal_line", "vertical_line",
        "half_vertical", "half_horizontal"
    ]

    regen_steps = 64
    fire_rate = 1.0  # no dropout at test time

    # ---- Run for Both Models and All Damage Types ----
    all_logs = {"classic_nca": {}, "graph_aug_nca": {}}
    for model_type, model, aug in [
        ("classic_nca", classic_nca, None),
        ("graph_aug_nca", graph_nca, graph_aug)
    ]:
        print(f"Testing {model_type}...")
        all_logs[model_type] = {}
        with torch.no_grad():
            # Grow to steady state
            seed = make_seed(n_channels, img_size, batch_size=1, device=device)
            state = seed.clone()
            for _ in range(regen_steps):
                if aug is not None:
                    state = model(state + aug(state), fire_rate=fire_rate)
                else:
                    state = model(state, fire_rate=fire_rate)
            grown_state = state.clone()  # [1, n_channels, H, W]

            for damage in damage_types:
                # Apply damage to RGBA in state
                damaged_state = apply_damage_to_state(grown_state, damage_type=damage)
                # Regenerate
                regen_states, metrics = rollout_regen(
                    model, damaged_state, fire_rate=fire_rate, steps=regen_steps, target=target, graph_aug=aug
                )
                # Save side-by-side image (damaged, regenerated)
                save_side_by_side(damaged_state[0, :4], regen_states[:, :4, :, :][0], damage, model_type, out_dir, upscale=4)
                # Save mean statistics
                metrics_arr = np.array([[m['mse'], m['ssim'], m['psnr'], m['pixel_perfection']] for m in metrics])
                mean_metrics = dict(
                    mse=float(metrics_arr[:,0].mean()),
                    ssim=float(metrics_arr[:,1].mean()),
                    psnr=float(metrics_arr[:,2].mean()),
                    pixel_perfection=float(metrics_arr[:,3].mean()),
                )
                all_logs[model_type][damage] = mean_metrics

    # ---- Save summary log (mean stats only) ----
    with open(os.path.join(out_dir, "regen_benchmark_results.json"), "w") as f:
        json.dump(all_logs, f, indent=2)
    print("All results saved in", out_dir)

if __name__ == "__main__":
    main()
