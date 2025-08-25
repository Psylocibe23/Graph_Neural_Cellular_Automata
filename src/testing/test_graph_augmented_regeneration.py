# src/testing/test_graph_attention_evolution.py
import os, re, glob, math, random, argparse, json, sys
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.nca_init import make_seed
from utils.damage import apply_damage_policy_
from modules.ncagraph import NeuralCAGraph

# ---------- utils ----------
def to_np(x):
    if isinstance(x, np.ndarray): return x
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def masked_rgb(x4, alpha_thr=0.1, upscale=1):
    if x4.ndim == 4: x4 = x4[0]
    rgb = x4[:3].detach().cpu()
    a   = x4[3:4].detach().cpu()
    m = (a > alpha_thr).float()
    rgb = rgb * m
    img = rgb.permute(1, 2, 0).numpy().clip(0, 1)
    if upscale > 1:
        t = torch.from_numpy(img).permute(2, 0, 1)[None]
        t = F.interpolate(t, scale_factor=upscale, mode="bilinear", align_corners=False)[0]
        img = t.permute(1, 2, 0).numpy().clip(0, 1)
    return img

def attn_to_rgb(a):
    a = to_np(a)
    vmax = np.percentile(a, 99.0) if a.max() <= 0 else a.max()
    a = np.clip(a / (vmax + 1e-8), 0, 1)
    cm = plt.get_cmap("viridis")
    return cm(a)[..., :3]

def save_combo(step, out_dir, rgb_img, attn, sender, receiver, group_maps):
    os.makedirs(out_dir, exist_ok=True)
    attn_np, sender_np, receiver_np = map(to_np, (attn, sender, receiver))
    rgb_np = to_np(rgb_img)
    rgb_map_np   = to_np(group_maps["rgb"])
    alpha_map_np = to_np(group_maps["alpha"])
    hidden_map_np= to_np(group_maps["hidden"])

    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(2, 3, 1); ax1.imshow(rgb_np); ax1.set_title("RGB (masked)"); ax1.axis("off")
    ax2 = plt.subplot(2, 3, 2); ax2.imshow(attn_np, cmap="viridis"); ax2.set_title("Graph attention"); ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3); ax3.imshow(attn_np, cmap="viridis")
    sender_rgba = np.zeros((*sender_np.shape,4), np.float32); sender_rgba[...,0]=1.0; sender_rgba[...,3]=(sender_np>0)*0.35
    ax3.imshow(sender_rgba)
    receiver_rgba = np.zeros((*receiver_np.shape,4), np.float32); receiver_rgba[...,1]=1.0; receiver_rgba[...,3]=(receiver_np>0)*0.35
    ax3.imshow(receiver_rgba)
    ax3.set_title("Attention + sender(magenta)/receiver(green)"); ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4); ax4.imshow(rgb_map_np, cmap="magma"); ax4.set_title("Per-group: RGB"); ax4.axis("off")
    ax5 = plt.subplot(2, 3, 5); ax5.imshow(alpha_map_np, cmap="magma"); ax5.set_title("Per-group: alpha"); ax5.axis("off")
    ax6 = plt.subplot(2, 3, 6); ax6.imshow(hidden_map_np, cmap="magma"); ax6.set_title("Per-group: hidden"); ax6.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"combo_{step:03d}.png")
    plt.savefig(path, dpi=140); plt.close(fig)
    return path

def force_damage_cfg(base_cfg, kind):
    """Build a config that guarantees a single specific damage kind is applied."""
    cfg = dict(base_cfg)  # shallow copy
    kinds = {k: 0.0 for k in base_cfg.get("kinds", {}).keys()}
    kinds[kind] = 1.0
    cfg.update({
        "start_epoch": 0,
        "prob": 1.0,
        "per_sample_prob": 1.0,
        "kinds": kinds
    })
    return cfg

def windows_to_wsl_path(p):
    # Convert "C:\Users\..." to "/mnt/c/Users/..."
    if len(p) >= 3 and p[1:3] == ":\\":
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gecko")
    ap.add_argument("--ckpt-path", required=True, help="Exact checkpoint .pt to load")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--damage-step", type=int, default=120)
    ap.add_argument("--fr", type=float, default=0.5)     # fixed fire rate for test
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-root", default=r"C:\Users\sprea\Desktop\pythonProject\GNN_NCA\outputs\graphaug_nca\test_regrowth")
    ap.add_argument("--kinds", default="", help="Comma list of damage kinds; default uses config.damage.kinds keys.")
    ap.add_argument("--include-clean", action="store_true", help="Also run a no-damage growth test.")
    args = ap.parse_args()

    # Normalize out path (works on Windows & WSL)
    out_root = windows_to_wsl_path(args.out_root)
    save_root = os.path.join(out_root, args.target)
    os.makedirs(save_root, exist_ok=True)

    config = load_config("configs/config.json")
    device     = config["misc"]["device"]
    n_ch       = int(config["model"]["n_channels"])
    img_size   = int(config["data"]["img_size"])
    alpha_thr  = float(config["model"].get("alpha_thr", 0.2))

    # ---- Build model with same knobs as training ----
    gcfg = config.get("graph_augmentation", {})
    model = NeuralCAGraph(
        n_channels=n_ch,
        update_hidden=int(config["model"]["update_mlp"]["hidden_dim"]),
        img_size=img_size,
        update_gain=float(config["model"].get("update_gain", 0.08)),
        alpha_thr=alpha_thr,
        use_groupnorm=bool(config["model"].get("use_groupnorm", True)),
        message_gain=float(gcfg.get("message_gain", 0.4)),
        hidden_only=bool(gcfg.get("hidden_only", True)),
        graph_d_model=int(gcfg.get("d_model", 16)),
        graph_attention_radius=int(gcfg.get("attention_radius", 5)),
        graph_num_neighbors=int(gcfg.get("num_neighbors", 16)),
        graph_gating_hidden=int(gcfg.get("gating_hidden", 32)),
        graph_zero_padded_shift=False,
        device=device,
    ).to(device).eval()

    # ---- Load the EXACT checkpoint you passed ----
    ckpt = torch.load(args.ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:   print(f"[test] missing keys: {missing}")
    if unexpected:print(f"[test] unexpected keys: {unexpected}")
    print(f"[test] Loaded {args.ckpt_path} (epoch {ckpt.get('epoch')})")

    # Determine which damage kinds to run
    base_damage = config.get("damage", {})
    if args.kinds.strip():
        damage_kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    else:
        damage_kinds = list(base_damage.get("kinds", {}).keys())

    # Optionally add a clean run
    if args.include_clean:
        damage_kinds = ["clean"] + damage_kinds

    # -------------- per-kind small test --------------
    for kind in damage_kinds:
        run_name = kind
        out_dir = os.path.join(save_root, run_name)
        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "checkpoint": os.path.abspath(args.ckpt_path),
            "epoch": int(ckpt.get("epoch")) if ckpt.get("epoch") is not None else None,
            "target": args.target,
            "damage_kind": kind,
            "damage_step": args.damage_step,
            "steps": args.steps,
            "fire_rate": args.fr
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # seed fresh state
        state = make_seed(n_ch, img_size, batch_size=1, device=device)

        attn_frames, combo_frames = [], []

        for t in range(args.steps + 1):
            # Apply the specific damage exactly once at t == damage_step
            if kind != "clean" and t == args.damage_step:
                cfg_specific = force_damage_cfg(base_damage, kind)
                with torch.no_grad():
                    apply_damage_policy_(state, cfg_specific, epoch=999999)
                print(f"[{kind}] applied damage at step {t}")

            pre_state = state.clone()

            with torch.no_grad():
                state, attn = model(pre_state, fire_rate=float(args.fr), return_attention=True)

            # sender/receiver masks (same dilation as training)
            with torch.no_grad():
                receiver = (F.max_pool2d(pre_state[:, 3:4], 3, 1, 1) > alpha_thr).float()[0, 0].cpu()
                sender   = receiver.clone()  # alive-to-alive
                # per-group magnitudes for visualization
                M = model.graph.msg_proj(pre_state)
                rgb_map   = M[0, :3].abs().mean(dim=0).cpu()
                alpha_map = M[0, 3:4].abs().mean(dim=0).cpu()
                hidden_map= M[0, 4:].abs().mean(dim=0).cpu()

            rgb_now = masked_rgb(state[:, :4], alpha_thr=alpha_thr, upscale=4)

            combo_png = save_combo(
                t, out_dir, rgb_now, attn[0].cpu(), sender, receiver,
                {"rgb": rgb_map, "alpha": alpha_map, "hidden": hidden_map}
            )

            # attention-only frames
            attn_rgb = attn_to_rgb(attn[0].cpu())
            plt.imsave(os.path.join(out_dir, f"attn_only_{t:03d}.png"), attn_rgb)

            attn_frames.append((attn_rgb * 255).astype(np.uint8))
            combo_frames.append((rgb_now * 255).astype(np.uint8))

            if t % 10 == 0:
                print(f"[{kind}] step {t:03d} -> {os.path.basename(combo_png)}")

        # Try to make MP4s
        try:
            import imageio.v3 as iio
            iio.imwrite(os.path.join(out_dir, "attention.mp4"), attn_frames, fps=args.fps)
            iio.imwrite(os.path.join(out_dir, "combo.mp4"), combo_frames, fps=args.fps)
            print(f"[{kind}] wrote attention.mp4 and combo.mp4 in {out_dir}")
        except Exception as e:
            print(f"[{kind}] [warn] imageio failed ({e}). PNGs are in {out_dir}.")
            print(f"ffmpeg -y -framerate {args.fps} -pattern_type glob -i '{out_dir}/attn_only_*.png' "
                  f"-c:v libx264 -pix_fmt yuv420p {os.path.join(out_dir, 'attention.mp4')}")
            print(f"ffmpeg -y -framerate {args.fps} -pattern_type glob -i '{out_dir}/combo_*.png' "
                  f"-c:v libx264 -pix_fmt yuv420p {os.path.join(out_dir, 'combo.mp4')}")

    print(f"All runs saved under: {save_root}")

if __name__ == "__main__":
    main()
