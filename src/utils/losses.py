import torch

def masked_loss(pred, target, alpha_thr: float = 0.2, lam_area: float = 5e-5):
    """
    Pixel loss masked by TARGET alpha (not by pred alpha), plus a tiny area penalty
    on predicted alpha to discourage background sprawl.

    Args:
        pred:   [B, 4, H, W] (RGBA from model)
        target: [B, 4, H, W] (RGBA target)
        alpha_thr: threshold for target 'alive' mask (slightly stricter than 0.1)
        lam_area: weight for alpha-area penalty (very small)

    Returns:
        per-sample loss vector [B]
    """
    # Supervise where the TARGET is alive, so the model is penalized for hallucinating elsewhere
    target_mask = (target[:, 3:4] > alpha_thr).float()  # [B,1,H,W]
    mse = ((pred - target) ** 2) * target_mask
    denom = target_mask.sum(dim=(1, 2, 3)) + 1e-8
    per_sample = mse.sum(dim=(1, 2, 3)) / denom

    # Small pressure to keep predicted alpha from flooding the canvas
    # (acts everywhere, not only inside target mask)
    area_pen = lam_area * pred[:, 3:4].mean(dim=(1, 2, 3))

    return per_sample + area_pen
