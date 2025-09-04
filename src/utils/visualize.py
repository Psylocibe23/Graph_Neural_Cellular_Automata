import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def save_comparison(target, pred, epoch, out_dir, filename=None, upscale=4):
    """
    Save a side-by-side comparison: target RGBA and predicted RGBA, optionally upscaled.
    """
    if hasattr(target, "detach"):
        target = target.detach().cpu()
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu()
    if target.ndim == 4:
        target = target[0]
    if pred.ndim == 4:
        pred = pred[0]

    # Optionally upscale for visualization
    def upscale_img(img, scale):
        if isinstance(img, torch.Tensor):
            if img.ndim == 2:  # [H, W]
                img = img.unsqueeze(0)  # [1, H, W]
            if img.ndim == 3:
                img = img.unsqueeze(0)  # [1, C, H, W]
            elif img.ndim == 4 and img.shape[0] != 1:
                img = img[0:1]
            img = torch.nn.functional.interpolate(
                img, scale_factor=scale, mode='bilinear', align_corners=False
            )
            img = img[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        else:
            from PIL import Image
            img = np.clip(img, 0, 1)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            img = pil_img.resize((img.shape[1] * scale, img.shape[0] * scale), Image.BILINEAR)
            img = np.asarray(img).astype(np.float32) / 255.0
        return img
    
    target_img = target[:4]
    pred_img = pred[:4]
    target_img = upscale_img(target_img, upscale)
    pred_img = upscale_img(pred_img, upscale)

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
