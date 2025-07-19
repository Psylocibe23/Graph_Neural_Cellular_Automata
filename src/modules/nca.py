import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.perception import FixedSobelPerception


class NeuralCA(nn.Module):
    def __init__(self, n_channels, update_hidden, update_layers=2, layer_norm=True, img_size=40, device="cpu"):
        super().__init__()
        self.n_channels = n_channels
        self.device = device

        # Perception (non-trainable Sobel + identity, depthwise)
        self.perception = FixedSobelPerception(n_channels)

        # Update rule: small MLP as 1x1 convs (configurable layers)
        layers = []
        in_dim = n_channels * 3  # after perception (identity, sobel_x, sobel_y per channel)
        for i in range(update_layers - 1):
            layers.append(nn.Conv2d(in_dim if i == 0 else update_hidden, update_hidden, 1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(update_hidden, n_channels, 1, bias=False))
        self.update_net = nn.Sequential(*layers)

        # LayerNorm over channels: H, W
        self.ln = nn.LayerNorm([n_channels, img_size, img_size]) if layer_norm else None


    def forward(self, x, fire_rate=1.0):
        y = self.perception(x)
        dx = self.update_net(y)
        if fire_rate < 1.0:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
            dx = dx * mask
        x = x + dx
        if self.ln:
            x = self.ln(x)
        return x
