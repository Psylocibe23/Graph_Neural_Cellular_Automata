import os, random, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.image import load_single_target_image
from utils.nca_init import make_seed

# Graph-augmented model
from modules.nca_graph import NeuralCAGraph


# ------------------------------------------------------------
# Utility: mask RGB by alpha and (optionally) upscale
# ------------------------------------------------------------
def masked_rgb(x4, alpha_thr=0.1, upscale=1):
    """
    x4: [1,4,H,W] or [4,H,W] (tensor)
    returns numpy [H,W,3] in [0,1]
    """
    if x4.ndim == 4:
        x4 = x4[0]
    rgb = x4[:3].detach().cpu()
    a   = x4[3:4].detach().cpu()
    m = (a > alpha_thr).float()
    rgb = rgb * m
    img = rgb.permute(1, 2, 0).numpy().clip(0, 1)

    if upscale > 1:
        t = torch.from_numpy(img).permute(2, 0, 1)[None]  # [1,3,H,W]
        t = F.interpolate(t, scale_factor=upscale, mode="bilinear", align_corners=False)[0]
        img = t.permute(1, 2, 0).numpy().clip(0, 1)
    return img


# ------------------------------------------------------------
# Zero-padded shift (to match the upgraded graph module)
# ------------------------------------------------------------
def shift2d(x, dy, dx, zero_pad=True):
    if not zero_pad:
        return torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    B, C, H, W = x.shape
    top, bot  = max(dy, 0), max(-dy, 0)
    left, right = max(dx, 0), max(-dx, 0)
    xpad = F.pad(x, (left, right, top, bot), value=0.0)
    return xpad[:, :, bot:bot + H, left:left + W]


# ------------------------------------------------------------
# Reconstruct the *exact* graph aggregation used this step
# (same offsets thanks to Python RNG state capture)
# Returns:
#   attn_map [H,W], sender_union [H,W], receiver_mask [H,W],
#   group_maps dict {"rgb":[..], "alpha":[..], "hidden":[..]},
#   per_offset (list of (dy,dx,map[H,W]))
# ------------------------------------------------------------
@torch.no_grad()
def debug_graph_from_state(x, graph_mod, alpha_thr=0.1):
    """
    x: [1,C,H,W] tensor BEFORE the update step
    graph_mod: modules.graph_augmentation.GraphAugmentation
    """
    B, C, H, W = x.shape
    assert B == 1, "This debug visual expects batch=1."

    # Pull config from the module
    d_model = graph_mod.d_model
    kmax    = graph_mod.num_neighbors
    offsets = graph_mod.offsets
    zero_pad = bool(getattr(graph_mod, "zero_padded_shift", True))
    alive_to_alive = bool(getattr(graph_mod, "alive_to_alive", True))
    temperature = graph_mod.scaling.abs().item() + 1e-6

    # Projections
    Q = graph_mod.query_proj(x)
    K = graph_mod.key_proj(x)
    M = graph_mod.msg_proj(x)

    Qp = Q.mean(dim=(2, 3))  # [1,d]
    if alive_to_alive:
        A_send = (F.max_pool2d(x[:, 3:4], 3, 1, 1) > alpha_thr).float()  # [1,1,H,W]

    # Receiver mask (same dilation as in the model)
    A_recv = (F.max_pool2d(x[:, 3:4], 3, 1, 1) > alpha_thr).float()  # [1,1,H,W]

    # Sample the same neighbor set (we restore RNG state outside)
    k = min(kmax, len(offsets))
    chosen = random.sample(offsets, k) if k > 0 else []

    messages = []
    logits   = []
    sender_masks = []

    for (dy, dx) in chosen:
        K_shift = shift2d(K, dy, dx, zero_pad)  # [1,d,H,W]
        M_shift = shift2d(M, dy, dx, zero_pad)  # [1,C,H,W]
        if alive_to_alive:
            src = shift2d(A_send, dy, dx, zero_pad)  # [1,1,H,W]
            M_shift = M_shift * src
            sender_masks.append(src)
        Kp = K_shift.mean(dim=(2, 3))            # [1,d]
        logit = (Qp * Kp).sum(dim=1)             # [1]
        logits.append(logit)
        messages.append(M_shift)

    if len(chosen) == 0:
        hw_zeros = torch.zeros(H, W, device=x.device)
        return hw_zeros, hw_zeros, A_recv[0, 0].cpu(), {"rgb": hw_zeros, "alpha": hw_zeros, "hidden": hw_zeros}, []

    L = torch.stack(logits, dim=0).squeeze(-1)       # [N]
    L = L - L.max()                                  # stabilize
    Wt = F.softmax(L / temperature, dim=0)           # [N]
    Wt_bc = Wt.view(len(chosen), 1, 1, 1, 1)         # [N,1,1,1,1]
    M_stack = torch.stack(messages, dim=0)           # [N,1,C,H,W]
    weighted = M_stack * Wt_bc                       # [N,1,C,H,W]

    # Attention map = mean over channels, sum over offsets
    attn_map = weighted.abs().mean(dim=2).sum(dim=0)[0, :, :]  # [H,W]

    # Sender union (alive_to_alive only)
    if alive_to_alive and sender_masks:
        src_stack = torch.stack(sender_masks, dim=0)  # [N,1,1,H,W]
        sender_union = (src_stack.max(dim=0).values)[0, 0]      # [H,W]
    else:
        sender_union = torch.zeros_like(attn_map)

    # Per-group maps (from weighted contributions BEFORE hidden-only policy)
    def ch_mean(slice_):
        return weighted[:, 0, slice_, :, :].abs().mean(dim=1).sum(dim=0)  # [H,W]

    rgb_map   = ch_mean(slice(0, 3)) if C >= 3 else torch.zeros_like(attn_map)
    alpha_map = ch_mean(slice(3, 4)) if C >= 4 else torch.zeros_like(attn_map)
    hidden_map= ch_mean(slice(4, C)) if C >  4 else torch.zeros_like(attn_map)

    # Per-offset contributions (for tiles)
    per_offset = []
    for i, (dy, dx) in enumerate(chosen):
        m_i = weighted[i, 0].abs().mean(dim=0)  # [H,W]
        per_offset.append(((dy, dx), m_i))

    return attn_map.cpu(), sender_union.cpu(), A_recv[0, 0].cpu(), \
           {"rgb": rgb_map.cpu(), "alpha": alpha_map.cpu(), "hidden": hidden_map.cpu()}, \
           per_offset


# ------------------------------------------------------------
# Plotting: main panel (generated RGB, attention, masks, groups)
# ------------------------------------------------------------
def save_main_panel(step, out_dir, rgb_img, attn, sender, receiver, group_maps):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(rgb_img)
    ax1.set_title("RGB (masked)")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(attn, cmap="viridis")
    ax2.set_title("Graph attention")
    ax2.axis("off")

    # Overlay: attention + sender/receiver
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(attn, cmap="viridis")
    # sender in magenta, receiver in cyan
    sender_rgb = np.zeros((*sender.shape, 4), dtype=np.float32)
    sender_rgb[..., 0] = 1.0  # R
    sender_rgb[..., 3] = (sender.numpy() > 0).astype(np.float32) * 0.35
    ax3.imshow(sender_rgb)
    receiver_rgb = np.zeros((*receiver.shape, 4), dtype=np.float32)
    receiver_rgb[..., 1] = 1.0  # G
    receiver_rgb[..., 3] = (receiver.numpy() > 0).astype(np.float32) * 0.35
    ax3.imshow(receiver_rgb)
    ax3.set_title("Attention + sender (magenta) / receiver (green)")
    ax3.axis("off")

    # Group maps
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(group_maps["rgb"], cmap="magma")
    ax4.set_title("Per-group: RGB")
    ax4.axis("off")

    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(group_maps["alpha"], cmap="magma")
    ax5.set_title("Per-group: alpha")
    ax5.axis("off")

    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(group_maps["hidden"], cmap="magma")
    ax6.set_title("Per-group: hidden")
    ax6.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"combo_{step:03d}.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------
# Plotting: per-offset tiles (directionality / border effects)
# ------------------------------------------------------------
def save_offset_tiles(step, out_dir, per_offset):
    if not per_offset:
        return None
    # choose tile grid size
    n = len(per_offset)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    for i, ((dy, dx), m) in enumerate(per_offset):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(m, cmap="viridis")
        ax.set_title(f"dy={dy}, dx={dx}")
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"offsets_{step:03d}.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------
# Main: roll the model and dump diagnostics every step
# ------------------------------------------------------------
def main():
    # --- settings ---
    config = load_config("configs/config.json")
    device      = config["misc"]["device"]
    n_channels  = int(config["model"]["n_channels"])
    img_size    = int(config["data"]["img_size"])
    alpha_thr   = float(config["model"].get("alpha_thr", 0.1))
    target_name = os.path.splitext(config["data"]["active_target"])[0]

    # Pick your checkpoint (graphaug run)
    ckpt_path = f"outputs/graphaug_nca/train_inter_loss/{target_name}/checkpoints/nca_epoch025.pt"
    out_dir   = f"outputs/graphaug_nca/test_attention/{target_name}"
    os.makedirs(out_dir, exist_ok=True)

    STEPS      = 120       # how many CA steps to roll
    UPSCALE    = 4         # upscale for the RGB view
    FR_RANDOM  = False     # use random fire rate like training
    FR_FIXED   = 0.5

    # --- build model ---
    model = NeuralCAGraph(
        n_channels=n_channels,
        update_hidden=int(config["model"]["update_mlp"]["hidden_dim"]),
        img_size=img_size,
        update_gain=float(config["model"].get("update_gain", 0.1)),
        alpha_thr=alpha_thr,
        use_groupnorm=bool(config["model"].get("use_groupnorm", True)),
        # graph knobs from config if present
        message_gain=float(config["graph_augmentation"].get("message_gain", 0.5)),
        hidden_only=bool(config["graph_augmentation"].get("hidden_only", True)),
        graph_d_model=int(config["graph_augmentation"].get("d_model", 16)),
        graph_attention_radius=int(config["graph_augmentation"].get("attention_radius", 4)),
        graph_num_neighbors=int(config["graph_augmentation"].get("num_neighbors", 8)),
        graph_gating_hidden=int(config["graph_augmentation"].get("gating_hidden", 32)),
        graph_alive_to_alive=bool(config["graph_augmentation"].get("alive_to_alive", True)),
        graph_zero_padded_shift=bool(config["graph_augmentation"].get("zero_padded_shift", True)),
        device=device,
    ).to(device)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:   print(f"[test] missing keys (initialized fresh): {missing}")
    if unexpected: print(f"[test] unexpected keys (ignored): {unexpected}")
    model.eval()

    # Seed
    state = make_seed(n_channels, img_size, batch_size=1, device=device)

    # Rollout with diagnostics
    for t in range(STEPS + 1):
        # ---- pre-step visuals will reflect the *next* transition ----
        pre_state = state.clone()

        # capture python RNG so graph offsets in debug match the forward call
        saved_rng = random.getstate()

        # step the model
        if FR_RANDOM:
            fr = float(torch.empty(1, device=device).uniform_(0.5, 1.0).item())
        else:
            fr = FR_FIXED

        with torch.no_grad():
            state, attn = model(pre_state, fire_rate=fr, return_attention=True)

        # reconstruct graph internals from the same RNG offsets on pre_state
        random.setstate(saved_rng)
        attn_dbg, senders, receivers, group_maps, per_offset = \
            debug_graph_from_state(pre_state, model.graph, alpha_thr=model.alpha_thr)

        # Prefer the attention returned by the model (already normalized).
        # If you want to cross-check, you can compare with attn_dbg.
        rgb_now = masked_rgb(state[:, :4], alpha_thr=alpha_thr, upscale=UPSCALE)

        # save panels
        main_path   = save_main_panel(t, out_dir, rgb_now, attn[0].cpu().numpy(), senders.numpy(), receivers.numpy(), 
                                      {k: v.numpy() for k, v in group_maps.items()})
        tiles_path  = save_offset_tiles(t, out_dir, per_offset)

        if t % 10 == 0:
            print(f"[test] saved {os.path.basename(main_path)}"
                  f"{' and ' + os.path.basename(tiles_path) if tiles_path else ''}")

    print(f"All diagnostics saved to {out_dir}")


if __name__ == "__main__":
    main()
