import torch
import torch.nn as nn


class FixedSobelPerception(nn.Module):
    """Non-trainable depthwise conv: identity + Sobel filters"""
    def __init__(self, n_channels):
        super().__init__()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        identity = torch.zeros((3,3), dtype=torch.float32)
        identity[1,1] = 1.0
        kernel = torch.stack([identity, sobel_x, sobel_y])
        kernel = kernel.unsqueeze(1)  # [3,1,3,3]
        weight = kernel.repeat(n_channels,1,1,1)  # [3*n_channels, 1, 3, 3]
        self.conv = nn.Conv2d(n_channels, n_channels*3, 3, 1, 1, groups=n_channels, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        y = self.conv(x)
        # [B, C*3, H, W] -> [B, 3, C, H, W] -> [B, 3*C, H, W]
        B, C3, H, W = y.shape
        y = y.view(B, -1, 3, H, W).permute(0,2,1,3,4).reshape(B, -1, H, W)
        return y
