import matplotlib.pyplot as plt
import numpy as np
import os

def save_comparison(target, pred, epoch, out_dir, filename=None):
    """
    Save a side-by-side comparison: target RGBA and predicted RGBA.

    Args:
        target: [4, H, W] or [B, 4, H, W] (torch.Tensor or numpy)
        pred:   [4, H, W] or [B, 4, H, W] (torch.Tensor or numpy)
        epoch:  int, current epoch
        out_dir: directory to save image
        filename: custom name (optional)
    """
    if hasattr(target, "detach"):
        target = target.detach().cpu()
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu()
    if target.ndim == 4:
        target = target[0]
    if pred.ndim == 4:
        pred = pred[0]

    # Clamp and transpose for plt.imshow
    target_img = target[:4].permute(1,2,0).numpy().clip(0,1)
    pred_img = pred[:4].permute(1,2,0).numpy().clip(0,1)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(target_img)
    axs[0].set_title("Target")
    axs[0].axis("off")
    axs[1].imshow(pred_img)
    axs[1].set_title("Predicted")
    axs[1].axis("off")
    plt.tight_layout()
    if filename is None:
        filename = f"comparison_epoch{epoch}.png"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)
