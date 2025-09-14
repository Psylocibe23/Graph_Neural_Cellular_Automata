import os, re, glob, json, time, random, sys, signal, shutil
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.config import load_config
from utils.image import load_single_target_image
from training.pool import SamplePool
from utils.nca_init import make_seed
from utils.utility_functions import count_parameters
from utils.visualize import save_comparison
from modules.ncagraph import NeuralCAGraph
from utils.damage import apply_damage_policy_  # <— NEW: unified damage policy


"""
Trainer for Graph-Augmented NCA.
- Uses full-canvas MSE on premultiplied RGBA (no background penalties).
- Random short/long rollouts, stochastic fire-rate, and sparse graph messages.
- Curriculum damage before rollouts; per-param grad-norm normalization.
- TensorBoard + image dumps; robust resume and checkpointing.
"""

# ------------------------- Loss (with background penalties) Non used here ------------------
def masked_loss(pred, target, alpha_thr=0.2,
                lam_area=5e-5, lam_bg_alpha=1e-3, lam_bg_rgb=2e-4):
    """
    pred, target: [B,4,H,W] in [0,1]
    - MSE where TARGET alpha > alpha_thr (on all 4 channels)
    - Background alpha penalty: pred alpha on TARGET-dead
    - Tiny background RGB penalty: pred RGB on TARGET-dead
    """
    tgt_alive = (target[:, 3:4] > alpha_thr).float()
    tgt_dead = 1.0 - tgt_alive

    mse = ((pred - target) ** 2) * tgt_alive
    denom = tgt_alive.sum(dim=(1, 2, 3)) + 1e-8
    primary = mse.sum(dim=(1, 2, 3)) / denom

    bg_alpha_pen = lam_bg_alpha * (pred[:, 3:4] * tgt_dead).mean(dim=(1, 2, 3))
    bg_rgb_pen = lam_bg_rgb   * (pred[:, :3]  * tgt_dead).abs().mean(dim=(1, 2, 3))
    area_pen = lam_area * pred[:, 3:4].mean(dim=(1, 2, 3))

    return primary + bg_alpha_pen + bg_rgb_pen + area_pen


def loss_premult_rgba(pred, target):
    """
    Full-canvas MSE on premultiplied RGBA.
    pred, target: [B,4,H,W] in [0,1]. RGB should be premultiplied by alpha.
    Returns per-sample loss [B].
    """
    # re-premultiply pred defensively 
    pred_rgb_prem = pred[:, :3] * pred[:, 3:4]
    pred_rgba = torch.cat([pred_rgb_prem, pred[:, 3:4]], dim=1)
    return F.mse_loss(pred_rgba, target, reduction="none").mean(dim=(1, 2, 3))


def save_grid_as_image(grid, filename):
    grid = grid.detach().cpu()
    if grid.ndim == 4:
        grid = grid[0]
    img = grid[:4].permute(1, 2, 0).numpy().clip(0, 1)
    plt.imsave(filename, img)

def extract_epoch_num(name):
    m = re.search(r'epoch(\d+)', name)
    return int(m.group(1)) if m else -1


# --- MAIN ---
def main():
    start_wall = time.time()
    config = load_config("configs/config.json")

    # ---- I/O ----
    target_name = os.path.splitext(config["data"]["active_target"])[0]
    base_dir = os.path.join("outputs", "graphaug_nca", "train_inter_loss", target_name)
    results_dir = os.path.join(base_dir, "images")
    inter_dir = os.path.join(base_dir, "intermediate")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    tb_dir = os.path.join(base_dir, "tb_logs")
    logs_dir = os.path.join(base_dir, "logs")
    for d in [results_dir, inter_dir, ckpt_dir, tb_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- seeds ----
    device = config["misc"]["device"]
    seed = int(config["misc"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---- Data ----
    target = load_single_target_image(config).to(device)  # [4,H,W]
    target[:3] = target[:3] * target[3:4]
    img_size = int(config["data"]["img_size"])
    n_ch = int(config["model"]["n_channels"])

    writer = SummaryWriter(log_dir=tb_dir)

    # ---- Seed funciton ----
    def seed_fn(batch_size=1):
        g = torch.zeros(batch_size, n_ch, img_size, img_size, device=device)
        cy = img_size // 2; cx = img_size // 2
        g[:, 3:4, cy, cx] = 1.0
        if n_ch > 4:
            g[:, 4:, cy, cx] = 0.01 * torch.randn_like(g[:, 4:, cy, cx])
        return g

    # ---- Model ----
    gcfg = config.get("graph_augmentation", {})
    model = NeuralCAGraph(
        n_channels=n_ch,
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
        graph_zero_padded_shift=False, 
        device=device,
    ).to(device)
    model.train()


    # --- Temporal sparsity controls for graph messages ---
    msg_rate = float(gcfg.get("message_rate", 1.0))
    msg_every = int(gcfg.get("message_every", 1)) 

    # ---- Optimizer / Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    scheduler_cfg = config["training"].get("scheduler")
    scheduler = None
    if isinstance(scheduler_cfg, dict):
        t = scheduler_cfg.get("type", "").lower()
        if t == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_cfg.get("step_size", 50)),
                gamma=float(scheduler_cfg.get("gamma", 0.7)),
            )
        elif t == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("t_max", 200)),
                eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
            )

    # ---- Pool ----
    pool = SamplePool(int(config["training"]["pool_size"]), seed_fn, device=device)

    # ---- Training knobs ----
    total_epochs = int(config["training"]["num_epochs"])
    steps_per_epoch = int(config["training"]["steps_per_epoch"])
    batch_size = int(config["training"]["batch_size"])
    short_min = int(config["training"]["nca_steps_min"])
    short_max = int(config["training"]["nca_steps_max"])
    long_prob = float(config["training"].get("long_rollout_prob", 0.25))
    long_min = int(config["training"].get("long_rollout_steps_min", 200))
    long_max = int(config["training"].get("long_rollout_steps_max", 400))
    fr_min = float(config["training"].get("fire_rate_min", 0.5))
    fr_max = float(config["training"].get("fire_rate_max", 1.0))

    # --- stability phase ---
    stab_enabled = bool(config["training"].get("stability_enabled", True))
    stab_K = int(config["training"].get("stability_K", 24))
    stab_thresh = float(config["training"].get("stability_threshold", 0.01))
    stab_weight = float(config["training"].get("stability_weight", 0.5))

    # --- loss knobs ---
    loss_alpha_thr = float(config["training"].get("loss_alpha_thr", 0.2))
    loss_lam_area = float(config["training"].get("loss_lam_area", 5e-5))
    loss_lam_bg_alpha = float(config["training"].get("loss_lam_bg_alpha", 1e-3))
    loss_lam_bg_rgb = float(config["training"].get("loss_lam_bg_rgb",   2e-4))

    damage_cfg = config.get("damage", {})

    # ---- Resume ----
    def _pick_resume(ckpt_dir):
        cand = []
    
        latest = os.path.join(ckpt_dir, "nca_latest.pt")
        if os.path.exists(latest): cand.append(latest)
    
        cand += sorted(glob.glob(os.path.join(ckpt_dir, "nca_epoch*_final.pt")))
        cand += sorted(glob.glob(os.path.join(ckpt_dir, "nca_*_last.pt")))
        cand += sorted(glob.glob(os.path.join(ckpt_dir, "nca_crash_ep*.pt")))
        cand += sorted(glob.glob(os.path.join(ckpt_dir, "nca_epoch*.pt")), key=extract_epoch_num)
    
        best_path, best_payload = None, None
        best_epoch, best_step = -1, -1
    
        for p in cand:
            try:
                payload = torch.load(p, map_location="cpu")
                ep = int(payload.get("epoch", -1))
                gs = int(payload.get("global_step", ep))
                if ep > best_epoch or (ep == best_epoch and gs > best_step):
                    best_epoch, best_step = ep, gs
                    best_path, best_payload = p, payload
            except Exception as _:
                continue
        return best_path, best_payload
    
    resume_path, resume_payload = _pick_resume(ckpt_dir)
    if resume_path is not None:
        missing, unexpected = model.load_state_dict(resume_payload["model_state"], strict=False)
        if missing: print(f"[resume] missing model keys: {missing}", flush=True)
        if unexpected:print(f"[resume] unexpected model keys: {unexpected}", flush=True)
        try:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        except Exception as e:
            print(f"[warn] optimizer state not compatible, reinitializing: {e}", flush=True)
        if scheduler is not None and resume_payload.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(resume_payload["scheduler_state"])
            except Exception as e:
                print(f"[warn] scheduler state not compatible, reinit: {e}", flush=True)
    
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        print(f"Resuming from {resume_path} (epoch {start_epoch-1})", flush=True)

    else:
        start_epoch = 1
        print("Starting training from scratch.", flush=True)


    # ---- Bookkeeping ----
    n_params = count_parameters(model)
    print(f"Params (graph NCA): {n_params}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training for {total_epochs} epochs")

    epoch_losses, pixel_scores, ssim_scores, psnr_scores = [], [], [], []
    last_epoch_finished = start_epoch - 1 

    terminate = {"flag": False}

    def _save_ckpt(tag: str, epoch_val: int, global_step_val: int):
        ckpt_path = os.path.join(ckpt_dir, f"nca_{tag}.pt")
        payload = {
            "epoch": epoch_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "param_count": n_params,
            "global_step": global_step_val,
        }
        torch.save(payload, ckpt_path)
        print(f"[ckpt] saved {ckpt_path}")

    def _handle_term(signum, frame):
        print(f"[signal] Received {signum}. Will save a LAST checkpoint and exit cleanly...")
        terminate["flag"] = True

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT,  _handle_term)

    # --- simple message-gain warmup schedule ---
    def scheduled_message_gain(ep, base=float(gcfg.get("message_gain", 0.5))):
        if ep < 100:  return 0.30
        if ep < 200:  return 0.40
        return base

    try:
        for epoch in trange(start_epoch, total_epochs + 1, desc="Epochs"):
            avg_loss = 0.0
            epoch_pixel, epoch_ssim, epoch_psnr = [], [], []

            base_gain_epoch = scheduled_message_gain(epoch)

            for step in trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
                idx, batch = pool.sample(batch_size)
                state = batch.to(device)

                # --- Damage curriculum ---
                apply_damage_policy_(state, damage_cfg, epoch)

                # --- rollout regime ---
                if random.random() < long_prob:
                    steps_lo, steps_hi = long_min, long_max
                else:
                    steps_lo, steps_hi = short_min, short_max

                nca_steps = torch.randint(steps_lo, steps_hi + 1, (batch_size,), device=device)
                max_steps = int(nca_steps.max().item())

                for t in range(max_steps):
                    mask = (nca_steps > t)
                    if not mask.any():
                        continue

                    fr = float(torch.empty(1, device=device).uniform_(fr_min, fr_max).item())

                    use_graph = True
                    if msg_every > 1:
                        use_graph = (t % msg_every == 0)
                    elif msg_rate < 1.0:
                        use_graph = (random.random() < msg_rate)

                    if hasattr(model, "message_gain"):
                        model.message_gain = base_gain_epoch if use_graph else 0.0

                    state[mask] = model(state[mask], fire_rate=fr)

                if hasattr(model, "message_gain"):
                    model.message_gain = base_gain_epoch

                # --- Loss ---
                """target_expand = target.unsqueeze(0).expand_as(state[:, :4])
                per_sample = masked_loss(
                    state[:, :4], target_expand,
                    alpha_thr=loss_alpha_thr,
                    lam_area=loss_lam_area,
                    lam_bg_alpha=loss_lam_bg_alpha,
                    lam_bg_rgb=loss_lam_bg_rgb
                )
                loss = per_sample.mean()"""
                # Loss
                target_expand = target.unsqueeze(0).expand_as(state[:, :4])
                per_sample = loss_premult_rgba(state[:, :4], target_expand)
                loss = per_sample.mean()

                # --- Stability phase ---
                """if stab_enabled:
                    with torch.no_grad():
                        close_mask = (per_sample < stab_thresh)
                    if close_mask.any():
                        state_stab = state.clone()
                        for k in range(stab_K):
                            frs = float(torch.empty(1, device=device).uniform_(fr_min, fr_max).item())
                            use_graph = True
                            if msg_every > 1:
                                use_graph = (k % msg_every == 0)
                            elif msg_rate < 1.0:
                                use_graph = (random.random() < msg_rate)
                            if hasattr(model, "message_gain"):
                                model.message_gain = base_gain_epoch if use_graph else 0.0
                            state_stab[close_mask] = model(state_stab[close_mask], fire_rate=frs)
                        if hasattr(model, "message_gain"):
                            model.message_gain = base_gain_epoch
                        stab = F.mse_loss(state_stab[close_mask, :4], target_expand[close_mask])
                        loss = loss + stab_weight * stab"""

                # --- Optimize ---
                """optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()"""
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Distill notebook trick: normalize each param's grad by its L2 norm
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.div_(p.grad.data.norm().add_(1e-8))

                optimizer.step()

                # --- Pool replace ---
                per_sample_detached = per_sample.detach()
                n_reset = int(float(config["training"].get("reset_worst_prob", 0.10)) * batch_size)
                worst_indices = torch.topk(per_sample_detached, n_reset).indices if n_reset > 0 else None
                do_random_reseed = (random.random() < float(config["training"].get("random_reseed_prob", 0.05)))
                rand_idx = torch.randint(0, batch_size, (1,), device=device).item() if do_random_reseed else None

                state_for_pool = state.detach()
                if (worst_indices is not None) or do_random_reseed:
                    state_for_pool = state_for_pool.clone()
                    if worst_indices is not None:
                        state_for_pool[worst_indices] = seed_fn(len(worst_indices)).detach()
                    if do_random_reseed:
                        state_for_pool[rand_idx:rand_idx+1] = seed_fn(1).detach()
                pool.replace(idx, state_for_pool)

                # --- Metrics ---
                """pred = state[:, :4]
                pred_img = pred[0].detach().cpu().numpy()
                target_img = target.cpu().numpy()
                diff = np.abs(pred_img[:4] - target_img[:4])
                pixel_perfect = (diff < 0.05).all(axis=0).mean()
                epoch_pixel.append(float(pixel_perfect))

                pred_np = pred[0, :3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                target_np = target[:3].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                epoch_ssim.append(ssim(pred_np, target_np, data_range=1.0, channel_axis=-1))
                epoch_psnr.append(psnr(pred_np, target_np, data_range=1.0))"""
                pred = state[:, :4]

                # Premultiply both for fair comparison
                pred_rgba = torch.cat([pred[:, :3] * pred[:, 3:4], pred[:, 3:4]], dim=1)
                tgt_rgba = target.unsqueeze(0).expand_as(pred)

                # Pixel-perfect on premultiplied RGBA
                pred_np_rgba = pred_rgba[0].detach().cpu().numpy()
                tgt_np_rgba = tgt_rgba[0].detach().cpu().numpy()
                diff = np.abs(pred_np_rgba - tgt_np_rgba)
                pixel_perfect = (diff < 0.05).all(axis=0).mean()
                epoch_pixel.append(float(pixel_perfect))

                # SSIM/PSNR on premultiplied RGB
                pred_np_rgb_prem = (pred_np_rgba[:3]).transpose(1, 2, 0).clip(0, 1)
                tgt_np_rgb_prem = (tgt_np_rgba[:3]).transpose(1, 2, 0).clip(0, 1)
                epoch_ssim.append(ssim(pred_np_rgb_prem, tgt_np_rgb_prem, data_range=1.0, channel_axis=-1))
                epoch_psnr.append(psnr(pred_np_rgb_prem,  tgt_np_rgb_prem,  data_range=1.0))


                avg_loss += loss.item()

                # --- Logging ---
                global_step = (epoch - 1) * steps_per_epoch + step
                writer.add_scalar('Loss/train', loss.item(), global_step)

                if (step + 1) % int(config["logging"]["visualize_interval"]) == 0:
                    img = pred[0, :3].detach().cpu().clamp(0, 1)
                    writer.add_image('Predicted/sample', img, global_step)
                    save_path0 = os.path.join(results_dir, f"epoch{epoch}_step{step+1}_sample0.png")
                    save_grid_as_image(state[0], save_path0)
                    save_comparison(target, pred[0], f"{epoch}_step{step+1}_sample0", results_dir, upscale=4)

                if (step + 1) % int(config["logging"]["log_interval"]) == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Epoch [{epoch}/{total_epochs}], Step [{step+1}/{steps_per_epoch}], "
                          f"Loss: {loss.item():.5f}")

                if terminate["flag"]:
                    last_epoch_finished = epoch 
                    _save_ckpt(f"ep{epoch}_step{step+1}_last", epoch, global_step)
                    writer.flush(); writer.close()
                    print("[signal] Last checkpoint saved. Exiting.")
                    sys.exit(0)

            # --- End of epoch ---
            avg_loss /= steps_per_epoch
            epoch_losses.append(avg_loss)
            pixel_scores.append(float(np.mean(epoch_pixel)))
            ssim_scores.append(float(np.mean(epoch_ssim)))
            psnr_scores.append(float(np.mean(epoch_psnr)))

            # --- row log ---
            with open(os.path.join(logs_dir, "training_log.jsonl"), "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "avg_loss": float(avg_loss),
                    "pixel_perfection": float(np.mean(epoch_pixel)),
                    "ssim": float(np.mean(epoch_ssim)),
                    "psnr": float(np.mean(epoch_psnr)),
                    "timestamp": datetime.now().isoformat()
                }) + "\n")

            writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch}] completed. "
                  f"Average loss: {avg_loss:.6f}")

            if scheduler is not None:
                scheduler.step()

            # --- Checkpoint ---
            ckpt_interval = int(config["logging"].get("checkpoint_interval_epochs", 25))
            if epoch % ckpt_interval == 0 or epoch == total_epochs:
                ckpt_payload = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "config": config,
                }
                ckpt_path = os.path.join(ckpt_dir, f"nca_epoch{epoch}.pt")
                torch.save(ckpt_payload, ckpt_path)
                torch.save(ckpt_payload, os.path.join(ckpt_dir, "nca_latest.pt"))  # rolling latest
                print(f"Model checkpoint saved to {ckpt_path} (and updated nca_latest.pt)")

            # also update rolling latest every epoch
            try:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "config": config,
                }, os.path.join(ckpt_dir, "nca_latest.pt"))
            except Exception as e:
                print(f"[warn] could not update nca_latest.pt: {e}")

            # mark last finished epoch
            last_epoch_finished = epoch

            if terminate["flag"]:
                global_step = epoch * steps_per_epoch
                _save_ckpt(f"epoch{epoch}_last", epoch, global_step)
                writer.flush(); writer.close()
                print("[signal] Last checkpoint saved at epoch boundary. Exiting.")
                sys.exit(0)

    except Exception as e:
        try:
            epoch_safe = locals().get("epoch", start_epoch - 1)
            step_safe  = locals().get("step", -1)
            global_step = (max(epoch_safe, 1) - 1) * steps_per_epoch + max(step_safe, 0)
            _save_ckpt(f"crash_ep{epoch_safe}_step{step_safe}", max(epoch_safe, 1), global_step)
            print(f"[crash] Saved emergency checkpoint due to: {e}")
        finally:
            raise

    # -------------------- AFTER TRAINING: always save a final checkpoint --------------------
    try:
        final_ckpt = os.path.join(ckpt_dir, f"nca_epoch{last_epoch_finished}_final.pt")
        torch.save({
            "epoch": last_epoch_finished,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
        }, final_ckpt)
        # also refresh latest
        shutil.copyfile(final_ckpt, os.path.join(ckpt_dir, "nca_latest.pt"))
        print(f"[ckpt] final checkpoint saved to {final_ckpt} (and updated nca_latest.pt)")
    except Exception as e:
        print(f"[warn] could not save final checkpoint: {e}")

    # -------------------- Summary JSON ---------------------------------
    log_summary = {
        "training_time_minutes": (time.time() - start_wall) / 60.0,
        "config": config,
        "parameter_count": n_params,
        "seed": config["misc"].get("seed"),
        "initial_loss": float(epoch_losses[0]) if epoch_losses else None,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else None,
        "epoch_losses": epoch_losses,
        "average_pixel_perfection": float(np.mean(pixel_scores)) if pixel_scores else None,
        "pixel_perfection_per_epoch": pixel_scores,
        "average_ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
        "ssim_per_epoch": ssim_scores,
        "average_psnr": float(np.mean(psnr_scores)) if psnr_scores else None,
        "psnr_per_epoch": psnr_scores
    }
    summary_path = os.path.join(logs_dir, f"training_log_ep{last_epoch_finished}.json")
    with open(summary_path, "w") as f:
        json.dump(log_summary, f, indent=2)
    print(f"Saved training log to {summary_path}")
    try:
        src = os.path.join(logs_dir, "training_log.jsonl")
        if os.path.exists(src):
            dst = os.path.join(logs_dir, f"training_log_ep{last_epoch_finished}.jsonl")
            shutil.copyfile(src, dst)
            print(f"Copied row log to {dst}")
    except Exception as e:
        print(f"[warn] could not copy jsonl log: {e}")

    writer.flush(); writer.close()


if __name__ == "__main__":
    main()
