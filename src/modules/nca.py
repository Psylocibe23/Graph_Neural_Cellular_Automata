import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.perception import FixedSobelPerception

class NeuralCA(nn.Module):
    def __init__(self, n_channels, update_hidden=128, img_size=40, layer_norm=True, device="cpu"):
        super().__init__()
        self.n_channels = n_channels
        self.device = device
        self.perception = FixedSobelPerception(n_channels)  # outputs [B, 48, H, W]

        in_dim = n_channels * 3  # 48 if n_channels=16
        layers = [
            nn.Conv2d(in_dim, update_hidden, 1),  # [B,48,H,W] -> [B,128,H,W]
            nn.ReLU(),
            nn.Conv2d(update_hidden, n_channels, 1, bias=False)  # [B,128,H,W] -> [B,16,H,W]
        ]
        self.update_net = nn.Sequential(*layers)
        nn.init.zeros_(self.update_net[-1].weight)  # zero-init last layer

        self.ln = nn.LayerNorm([n_channels, img_size, img_size]) if layer_norm else None

    def forward(self, x, fire_rate=1.0):
        y = self.perception(x)  # [B, 48, H, W]
        dx = self.update_net(y)  # [B, 16, H, W]
        if fire_rate < 1.0:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
            dx = dx * mask
        x = x + dx
        # Alive masking: set ALL channels to zero if not in 3x3 neighborhood of alive cell
        alive_mask = (F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1).float()
        x = x * alive_mask  # [B, 16, H, W]
        if self.ln:
            x = self.ln(x)
        return x
