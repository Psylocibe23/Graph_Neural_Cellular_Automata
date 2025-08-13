# src/testing/test_graphaug_growth.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from utils.nca_init import make_seed
from utils.config import load_config
from utils.image import load_single_target_image

# Graph-augmented NCA class
from modules.ncagraph import NeuralCAGraph


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def save_img(img, path, upscale=4):
    """
    Save RGBA image, upscaled, masking out dead cells (alpha < 0.1).
    img can be [C,H,W] or [1,C,H,W] or a tensor; we clamp and save with plt.
    """
    if hasattr(img, "detach"):
        img = img.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img_np = img[:4].permute(1, 2, 0).numpy().copy()  # [H,W,4]
    alpha = img_np[..., 3:4]
    mask = (alpha > 0.1).astype(np.float32)
    img_np[..., :3] *= mask
    img_np[..., 3:] = mask

    if upscale > 1:
        img_t = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]
        img_t = F.interpolate(img_t, scale_factor=upscale, mode='bilinear', align_corners=False)[0]
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

    assert img_np.ndim == 3 and img_np.shape[2] in (3, 4), f"Got shape {img_np.shape}"
    plt.imsave(path, img_np)


def save_attn(attn_hw, path, upscale=4, cmap='viridis'):
    """
    Save an attention heatmap from a [H,W] (or [1,H,W]) tensor/ndarray in [0,1].
    """
    if hasattr(attn_hw, "detach"):
        attn_hw = attn_hw.detach().cpu()
    if isinstance(attn_hw, torch.Tensor):
        if attn_hw.ndim == 3 and attn_hw.shape[0] == 1:
            attn_hw = attn_hw[0]
        attn_np = attn_hw.numpy()
    else:
        attn_np = np.asarray(attn_hw)

    # Normalize robustly to [0,1]
    a_min, a_max = float(attn_np.min()), float(attn_np.max())
    if a_max > a_min:
        attn_np = (attn_np - a_min) / (a_max - a_min)
    else:
        attn_np = np.zeros_like(attn_np)

    # Upscale via torch for consistency
    if upscale > 1:
        t = torch.tensor(attn_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        t = F.interpolate(t, scale_factor=upscale, mode='bilinear', align_corners=False)[0, 0]
        attn_np = t.numpy()

    plt.imsave(path, attn_np, cmap=cmap, vmin=0.0, vmax=1.0)


def save_side_by_side(rgb_img, attn_hw, path, upscale=4, cmap='viridis', title_left='RGB', title_right='Attention'):
    """
    Make a side-by-side figure: masked RGB on the left, attention heatmap on the right.
    """
    # Prepare RGB (masked)
    if hasattr(rgb_img, "detach"):
        rgb_img = rgb_img.detach().cpu()
    if rgb_img.ndim == 4:
        rgb_img = rgb_img[0]
    rgb_np = rgb_img[:4].permute(1, 2, 0).numpy().copy()
    alpha = rgb_np[..., 3:4]
    mask  = (alpha > 0.1).astype(np.float32)
    rgb_np[..., :3] *= mask
    rgb_np = np.clip(rgb_np[..., :3], 0, 1)

    # Prepare attention
    if hasattr(attn_hw, "detach"):
        attn_hw = attn_hw.detach().cpu()
    if isinstance(attn_hw, torch.Tensor):
        if attn_hw.ndim == 3 and attn_hw.shape[0] == 1:
            attn_hw = attn_hw[0]
        attn_np = attn_hw.numpy()
    else:
        attn_np = np.asarray(attn_hw)

    # Normalize [0,1]
    a_min, a_max = float(attn_np.min()), float(attn_np.max())
    if a_max > a_min:
        attn_np = (attn_np - a_min) / (a_max - a_min)
    else:
        attn_np = np.zeros_like(attn_np)

    # Upscale both for nicer viewing
    if upscale > 1:
        # RGB
        t_rgb = torch.tensor(rgb_np).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        t_rgb = F.interpolate(t_rgb, scale_factor=upscale, mode='bilinear', align_corners=False)[0]
        rgb_np = t_rgb.permute(1, 2, 0).numpy()
        rgb_np = np.clip(rgb_np, 0, 1)
        # ATTN
        t_a = torch.tensor(attn_np).unsqueeze(0).unsqueeze(0)
        t_a = F.interpolate(t_a, scale_factor=upscale, mode='bilinear', align_corners=False)[0, 0]
        attn_np = t_a.numpy()

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(rgb_np)
    axs[0].set_title(title_left)
    axs[1].imshow(attn_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axs[1].set_title(title_right)
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # --- SETTINGS ---
    config = load_config('configs/config.json')
    device      = config["misc"]["device"]
    n_channels  = int(config["model"]["n_channels"])
    img_size    = int(config["data"]["img_size"])
    steps       = 400
    upscale     = 4
    target_name = os.path.splitext(config["data"]["active_target"])[0]

    # Pick your checkpoint (graph-aug run tree)
    ckpt_path = f"outputs/graphaug_nca/train_inter_loss/{target_name}/checkpoints/nca_epoch20.pt"
    save_dir  = f"outputs/graphaug_nca/test_growth/{target_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Optional: load target for your own inspection
    _ = load_single_target_image(config).to(device)

    # --- BUILD MODEL (mirror training args) ---
    gcfg = config.get("graph_augmentation", {})
    model = NeuralCAGraph(
        n_channels=n_channels,
        update_hidden=int(config["model"]["update_mlp"]["hidden_dim"]),
        img_size=img_size,
        update_gain=float(config["model"].get("update_gain", 0.1)),
        alpha_thr=float(config["model"].get("alpha_thr", 0.1)),
        use_groupnorm=bool(config["model"].get("use_groupnorm", True)),
        # Graph knobs
        message_gain=float(gcfg.get("message_gain", 0.5)),
        hidden_only=bool(gcfg.get("hidden_only", True)),
        graph_d_model=int(gcfg.get("d_model", 16)),
        graph_attention_radius=int(gcfg.get("attention_radius", 4)),
        graph_num_neighbors=int(gcfg.get("num_neighbors", 8)),
        graph_gating_hidden=int(gcfg.get("gating_hidden", 32)),
        device=device
    ).to(device)

    # Load weights (strict=False for robustness)
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[test] missing model keys (initialized fresh): {missing}")
    if unexpected:
        print(f"[test] unexpected model keys (ignored): {unexpected}")
    model.eval()

    # --- PREPARE SEED ---
    seed = make_seed(n_channels, img_size, batch_size=1, device=device)

    # --- ROLLOUT ---
    state = seed.clone()
    all_imgs  = []
    all_attns = []

    # Fire-rate policy (match train): fixed 0.5 or random in [0.5,1.0]
    FR_FIXED  = 0.5
    FR_RANDOM = False

    # For visualization, we typically want graph messages every step
    # If your class exposes a message sparsity knob, keep it on; else just run.
    with torch.no_grad():
        for t in range(steps + 1):
            # Save current frame (before update)
            frame_path = os.path.join(save_dir, f"frame_{t:03d}.png")
            save_img(state[:, :4], frame_path, upscale=upscale)
            all_imgs.append(state[:, :4][0].cpu())

            # Try to step while ALSO retrieving the attention map used at this step
            fr = float(torch.empty(1, device=device).uniform_(0.5, 1.0).item()) if FR_RANDOM else FR_FIXED

            attn_map = None
            # Preferred path: model(..., return_attention=True) returns (state_next, attn)
            try:
                state, attn_map = model(state, fire_rate=fr, return_attention=True)
            except TypeError:
                # Fallback #1: some classes expose an internal graph module
                try:
                    # Step once without attention to get next state
                    state = model(state, fire_rate=fr)
                    # Now ask the graph to compute attn on the pre-update state or on current state
                    if hasattr(model, "graph"):
                        # Many GraphAug impls accept return_attention_map=True
                        _maybe, attn_map = model.graph(state, return_attention_map=True)
                        # Some return (gated_msg, attn); some return just attn. Handle both:
                        if isinstance(_maybe, torch.Tensor) and _maybe.ndim >= 3:
                            # first is gated message, second is attn
                            pass
                        else:
                            # _maybe was actually the attn; adjust
                            attn_map = _maybe
                except Exception:
                    # Fallback #2: if attention is unavailable, keep None
                    pass

            # Save attention (if available)
            if attn_map is not None:
                # Expect attn_map shape [B,H,W] or [H,W]
                attn_hw = attn_map[0] if (isinstance(attn_map, torch.Tensor) and attn_map.ndim == 3) else attn_map
                attn_path = os.path.join(save_dir, f"attn_{t:03d}.png")
                save_attn(attn_hw, attn_path, upscale=upscale, cmap='viridis')

                # Side-by-side composite
                combo_path = os.path.join(save_dir, f"combo_{t:03d}.png")
                save_side_by_side(state[:, :4], attn_hw, combo_path, upscale=upscale,
                                  title_left='RGB (masked)', title_right='Graph Attention')

                all_attns.append(attn_hw.detach().cpu() if isinstance(attn_hw, torch.Tensor) else torch.tensor(attn_hw))
            else:
                all_attns.append(None)

    print(f"All frames saved to {save_dir}")
    print(f"Example files:\n - {os.path.join(save_dir, 'frame_000.png')}\n - {os.path.join(save_dir, 'attn_000.png')} (if attention available)\n - {os.path.join(save_dir, 'combo_000.png')} (side-by-side)")

    # --- OPTIONAL: quick montage of selected frames (RGB only) ---
    select_steps = [1, 2, 4, 8, 16, 32, 49]
    n = len(select_steps)
    plt.figure(figsize=(n * 2, 2.5))
    for i, step in enumerate(select_steps):
        plt.subplot(1, n, i + 1)
        img = all_imgs[step][:3].permute(1, 2, 0).numpy().clip(0, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Step {step}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
