import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class GraphAugmentation(nn.Module):
    """
    Augments local NCA with mid-range graph message passing, gated channelwise 
    """
    def __init__(self, n_channels, d_model=16, attention_radius=4, num_neighbors=8, gating_hidden=32):
        """
        n_channels: Number of channels in NCA state
        d_model: Dimension of projected queries/keys
        attention_radius: Maximum distance for "mid-range" neighbors
        num_neighbors: Number of neighbors to attend to (sampled each step)
        gating_hidden: Hidden size for channel-wise gate MLP
        """
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.attention_radius = attention_radius
        self.num_neighbors = num_neighbors

        # Q, K, M projections: 1x1 convs for each
        self.query_proj = nn.Conv2d(n_channels, d_model, 1)
        self.key_proj = nn.Conv2d(n_channels, d_model, 1)
        self.msg_proj = nn.Conv2d(n_channels, n_channels, 1)  # messages keep all channels

        # Learnable scaling (alpha)
        self.scaling = nn.Parameter(torch.tensor(math.sqrt(d_model), dtype=torch.float32))

        # Channel-wise gate
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(n_channels*2, gating_hidden, 1),
            nn.ReLU(),
            nn.Conv2d(gating_hidden, n_channels, 1),
            nn.Sigmoid()
        )

        # Build all possible mid-range offsets (excluding zero/center and local 1-step neighbors)
        self.offsets = self._build_offsets(attention_radius)

    def _build_offsets(self, radius):
        offsets = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if (dx, dy) == (0, 0): continue  # skip self
                if abs(dx) <= 1 and abs(dy) <= 1: continue  # skip local neighbors
                offsets.append((dy, dx))
        return offsets

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device

        # Project Q, K, M
        Q = self.query_proj(x)  # [B, d_model, H, W]
        K = self.key_proj(x)  # [B, d_model, H, W]
        M = self.msg_proj(x)  # [B, C, H, W]

        # Spatial mean-pooling: [B, d_model, H, W] -> [B, d_model]
        Q_pooled = Q.mean(dim=[2,3])  # [B, d_model]
        K_pooled = K.mean(dim=[2,3])  # [B, d_model]

        # For each cell, gather mid-range neighbors (randomly sample num_neighbors)
        chosen_offsets = random.sample(self.offsets, min(self.num_neighbors, len(self.offsets)))
        messages = []
        affinities = []

        for dy, dx in chosen_offsets:
            shifted_K = torch.roll(K, shifts=(dy, dx), dims=(2,3))
            shifted_M = torch.roll(M, shifts=(dy, dx), dims=(2,3))
            shifted_K_pooled = shifted_K.mean(dim=[2,3])  # [B, d_model]
            # Compute attention affinity: (q_i^T k_j) / alpha
            attn = (Q_pooled * shifted_K_pooled).sum(dim=1) / (self.scaling + 1e-6)  # [B]
            attn = attn.view(B, 1, 1, 1)  # broadcast
            affinities.append(attn)
            messages.append(shifted_M)

        # Stack and softmax over neighbors
        messages = torch.stack(messages, dim=0)  # [num_neighbors, B, C, H, W]
        affinities = torch.stack(affinities, dim=0)  # [num_neighbors, B, 1, 1, 1]
        attn_weights = F.softmax(affinities, dim=0)  # [num_neighbors, B, 1, 1, 1]

        # Weighted sum of messages
        agg_message = (messages * attn_weights).sum(dim=0)  # [B, C, H, W]

        # Channel-wise gate (learns for each channel how much to use the mid-range info)
        concat = torch.cat([x, agg_message], dim=1)  # [B, 2C, H, W]
        gate = self.gate_mlp(concat)  # [B, C, H, W], [0,1]
        gated_message = agg_message * gate

        return gated_message  # [B, C, H, W]
