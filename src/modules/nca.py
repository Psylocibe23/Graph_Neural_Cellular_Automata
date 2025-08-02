import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.perception import FixedSobelPerception

class NeuralCA(nn.Module):
    def __init__(self, n_channels, update_hidden, update_layers=2, layer_norm=True, img_size=40, device="cpu"):
        super().__init__()
        self.n_channels = n_channels
        self.device = device
        self.perception = FixedSobelPerception(n_channels)

        # Update rule: small MLP as 1x1 convs (configurable layers)
        layers = []
        in_dim = n_channels * 3  # after perception (identity, sobel_x, sobel_y per channel)
        for i in range(update_layers - 1):
            layers.append(nn.Conv2d(in_dim if i == 0 else update_hidden, update_hidden, 1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(update_hidden, n_channels, 1, bias=False))
        layers.append(nn.Tanh())  # Essential for stable gradual updates!
        self.update_net = nn.Sequential(*layers)

        self.ln = nn.LayerNorm([n_channels, img_size, img_size]) if layer_norm else None

    def alive_mask(self, x, alpha_thresh=0.1):
        # Returns binary mask where alpha > threshold
        return (x[:, 3:4, :, :] > alpha_thresh).float()

    def neighbor_alive_mask(self, x, alpha_thresh=0.1):
        # Returns binary mask where any neighbor (3x3) is alive
        mask = self.alive_mask(x, alpha_thresh)
        kernel = torch.ones(1, 1, 3, 3, device=x.device)
        # Convolve alive mask to count alive neighbors
        neighbor_count = F.conv2d(mask, kernel, padding=1)
        return (neighbor_count > 0).float()  # 1 if any neighbor is alive

    def forward(self, x, fire_rate=1.0):
        y = self.perception(x)
        dx = self.update_net(y)
        if fire_rate < 1.0:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
            dx = dx * mask
        x = x + dx
        # Don't do in-place edits; operate on slices then re-assemble
        rgb = x[:, :3, :, :]
        alpha = x[:, 3:4, :, :]
    
        # Clamp/relu the alpha channel, then propagate only from alive neighbors
        alpha = torch.clamp(alpha, 0, 1)
        neighbor_mask = self.neighbor_alive_mask(x)
        alpha = alpha * neighbor_mask
        alpha = alpha * (alpha > 0.1).float()
        # zero-out RGB for dead cells
        rgb = rgb * (alpha > 0.1).float()
        # Re-assemble the tensor
        x = torch.cat([rgb, alpha, x[:, 4:, :, :]], dim=1) if x.shape[1] > 4 else torch.cat([rgb, alpha], dim=1)
        if self.ln:
            x = self.ln(x)
        return x
