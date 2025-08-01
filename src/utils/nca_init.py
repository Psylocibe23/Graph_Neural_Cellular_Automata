import torch


def make_seed(n_channels, img_size, batch_size=1, device="cpu"):
    grid = torch.zeros(batch_size, n_channels, img_size, img_size, device=device)
    grid[:, 3, img_size//2, img_size//2] = 1.0  # Only alpha at center
    return grid