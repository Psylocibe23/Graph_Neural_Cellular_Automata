import math, random
import torch
import torch.nn.functional as F

"""
utils.damage — in-place damage operators + config-driven policy for NCA training/tests.
  - cutout_square_, cutout_circle_, stripe_wipe_ → hard deletions (all channels = 0)
  - alpha_dropout_, salt_pepper_alpha_  → alpha-only deletions (shape-aware/agnostic)
  - hidden_scramble_ → noise on hidden channels only (keeps alpha)
  - gaussian_hole_ → soft radial “burn” (multiplicative damp)
  - apply_damage_policy_ → reads a dict config and applies ONE sampled damage kind to the whole batch, in-place.

Config keys (with back-compat fallbacks)
  start_epoch / damage_start_epoch: begin applying damage after this epoch
  prob / damage_prob: chance to apply any damage on a given call
  kinds: dict of {name: weight} to sample damage type
  size_min / size_max / damage_patch_size : size range (pixels) for geometric damages
  stripe_width: band width for stripe_wipe_
  alpha_thr: alive threshold for alpha-conditioned ops
  alpha_dropout_p: drop probability in alpha_dropout_
  salt_pepper_p: per-pixel zeroing prob for salt_pepper_alpha_
  hidden_noise_sigma: std for hidden_scramble_
  gaussian_softness: edge softness for gaussian_hole_
"""


@torch.no_grad()
def cutout_square_(state: torch.Tensor, size: int):
    """Zero all channels in a random square of side size."""
    B, C, H, W = state.shape
    if size <= 0: return
    for b in range(B):
        y = int(torch.randint(0, max(1, H - size + 1), (1,), device=state.device))
        x = int(torch.randint(0, max(1, W - size + 1), (1,), device=state.device))
        state[b, :, y:y+size, x:x+size] = 0.0

@torch.no_grad()
def cutout_circle_(state: torch.Tensor, radius: int):
    """Zero all channels inside a random circle of radius R."""
    B, C, H, W = state.shape
    if radius <= 0: return
    yy = torch.arange(H, device=state.device).view(H, 1).float()
    xx = torch.arange(W, device=state.device).view(1, W).float()
    for b in range(B):
        cy = int(torch.randint(radius, max(radius+1, H - radius), (1,), device=state.device))
        cx = int(torch.randint(radius, max(radius+1, W - radius), (1,), device=state.device))
        mask = ((yy - cy)**2 + (xx - cx)**2) <= (radius**2)
        state[b, :, mask] = 0.0

@torch.no_grad()
def stripe_wipe_(state: torch.Tensor, width: int, orientation: str = "auto"):
    """Zero a random horizontal or vertical band of width width."""
    B, C, H, W = state.shape
    if width <= 0: return
    if orientation == "auto":
        orientation = "h" if random.random() < 0.5 else "v"
    if orientation == "h":
        y0 = int(torch.randint(0, max(1, H - width + 1), (1,), device=state.device))
        state[:, :, y0:y0+width, :] = 0.0
    else:
        x0 = int(torch.randint(0, max(1, W - width + 1), (1,), device=state.device))
        state[:, :, :, x0:x0+width] = 0.0

@torch.no_grad()
def alpha_dropout_(state: torch.Tensor, p: float, alpha_thr: float = 0.1, hard: bool = True):
    """
    Randomly kill a fraction p of currently-alive alpha pixels.
    hard=True -> zero all channels at dropped pixels; else only alpha.
    """
    if p <= 0: return
    alpha = state[:, 3:4]
    alive = (alpha > alpha_thr).float()
    drop = (torch.rand_like(alpha) < p).float() * alive
    if hard:
        state *= (1.0 - drop)  # zero everything where we drop
    else:
        state[:, 3:4] = alpha * (1.0 - drop)

@torch.no_grad()
def salt_pepper_alpha_(state: torch.Tensor, p: float):
    """Sparse alpha ‘pepper’: randomly zero alpha (not conditioned on alive)."""
    if p <= 0: return
    mask = (torch.rand_like(state[:, 3:4]) < p).float()
    state[:, 3:4] *= (1.0 - mask)

@torch.no_grad()
def hidden_scramble_(state: torch.Tensor, sigma: float = 0.2):
    """Add noise only to hidden channels (>=4); does not change alpha."""
    B, C, H, W = state.shape
    if C <= 4 or sigma <= 0: return
    noise = torch.randn(B, C-4, H, W, device=state.device) * sigma
    state[:, 4:] = (state[:, 4:] + noise).clamp_(0.0, 1.0)

@torch.no_grad()
def gaussian_hole_(state: torch.Tensor, radius: int, softness: float = 0.35):
    """
    Soft ‘burn’: multiply all channels by (1 - soft disk).
    radius sets extent; softness sets edge steepness.
    """
    B, C, H, W = state.shape
    if radius <= 0: return
    yy = torch.arange(H, device=state.device).view(H, 1).float()
    xx = torch.arange(W, device=state.device).view(1, W).float()
    for b in range(B):
        cy = int(torch.randint(radius, max(radius+1, H - radius), (1,), device=state.device))
        cx = int(torch.randint(radius, max(radius+1, W - radius), (1,), device=state.device))
        r2 = (yy - cy)**2 + (xx - cx)**2
        mask = torch.exp(-(r2 / (2.0 * (radius * max(1e-6, softness))**2)))
        damp = (1.0 - mask).clamp(0.0, 1.0)  # 0 at center, ~1 outside
        state[b] *= damp

@torch.no_grad()
def apply_damage_policy_(state: torch.Tensor, dmg_cfg: dict, epoch: int):
    """
    Read config and apply one sampled damage in-place to state.
    state: [B,C,H,W] 
    """
    # Back-compat with your previous config keys
    start_ep = int(dmg_cfg.get("start_epoch", dmg_cfg.get("damage_start_epoch", 100)))
    prob = float(dmg_cfg.get("prob", dmg_cfg.get("damage_prob", 0.0)))
    if epoch < start_ep or prob <= 0: 
        return
    if torch.rand(1, device=state.device).item() > prob:
        return

    kinds = dmg_cfg.get("kinds", {"square": 1.0})
    names, weights = zip(*kinds.items())
    kind = random.choices(names, weights=weights, k=1)[0]

    size_min = int(dmg_cfg.get("size_min", dmg_cfg.get("damage_patch_size", 8)))
    size_max = int(dmg_cfg.get("size_max", max(size_min, 14)))
    size = int(random.randint(size_min, size_max))

    alpha_thr = float(dmg_cfg.get("alpha_thr", 0.1))
    alpha_drop_p = float(dmg_cfg.get("alpha_dropout_p", 0.1))
    stripe_width = int(dmg_cfg.get("stripe_width", size))
    saltpepper_p = float(dmg_cfg.get("salt_pepper_p", 0.02))
    hidden_sigma = float(dmg_cfg.get("hidden_noise_sigma", 0.0))
    gaussian_soft = float(dmg_cfg.get("gaussian_softness", 0.35))

    if kind == "square": cutout_square_(state, size)
    elif kind == "circle": cutout_circle_(state, size//2 if size>1 else 1)
    elif kind == "stripes": stripe_wipe_(state, stripe_width, orientation="auto")
    elif kind == "alpha_drop": alpha_dropout_(state, alpha_drop_p, alpha_thr=alpha_thr, hard=True)
    elif kind == "saltpepper": salt_pepper_alpha_(state, saltpepper_p)
    elif kind == "gaussian": gaussian_hole_(state, radius=max(1, size//2), softness=gaussian_soft)
    elif kind == "hidden_noise": hidden_scramble_(state, sigma=hidden_sigma)
    else:
        # default fallback
        cutout_square_(state, size)
