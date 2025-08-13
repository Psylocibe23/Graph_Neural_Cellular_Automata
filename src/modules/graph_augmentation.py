import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAugmentation(nn.Module):
    """
    Mid-range, gated message passing used by NeuralCAGraph.

    Computes per-pixel messages by aggregating features from a sparse set of
    mid-range offsets using attention weights. Optional upgrades:
      • alive_to_alive: only 'alive' (alpha-dilated) source cells send messages
      • zero_padded_shift: use zero-padded shifts instead of torch.roll to avoid wrap

    Returns either just the aggregated message, or (message, attention_map) when
    return_attention_map=True (useful for visualization).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 16,
        attention_radius: int = 4,
        num_neighbors: int = 8,
        gating_hidden: int = 32,
        *,
        alive_to_alive: bool = True,
        zero_padded_shift: bool = True,
        alpha_thr: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.attention_radius = attention_radius
        self.num_neighbors = num_neighbors
        self.alive_to_alive = bool(alive_to_alive)
        self.zero_padded_shift = bool(zero_padded_shift)
        self.alpha_thr = float(alpha_thr)

        # 1x1 projections
        self.query_proj = nn.Conv2d(n_channels, d_model, 1)
        self.key_proj   = nn.Conv2d(n_channels, d_model, 1)
        self.msg_proj   = nn.Conv2d(n_channels, n_channels, 1)  # messages keep all channels

        # Learnable temperature (denominator) for logits; init ~ sqrt(d_model)
        self.scaling = nn.Parameter(torch.tensor(math.sqrt(d_model), dtype=torch.float32))

        # Channel-wise gate: decides per channel how much of agg_message to use
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(n_channels * 2, gating_hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(gating_hidden, n_channels, 1),
            nn.Sigmoid(),
        )

        # Precompute list of mid-range offsets (exclude self and 3x3 locals)
        self.offsets = self._build_offsets(attention_radius)

    @staticmethod
    def _build_offsets(radius: int):
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dy, dx) == (0, 0):
                    continue
                if abs(dy) <= 1 and abs(dx) <= 1:  # remove local neighborhood
                    continue
                offsets.append((dy, dx))
        return offsets

    @staticmethod
    def _shift2d_pad(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Zero-padded shift (no wrap)."""
        B, C, H, W = x.shape
        top, bot  = max(dy, 0), max(-dy, 0)
        left, right = max(dx, 0), max(-dx, 0)
        xpad = F.pad(x, (left, right, top, bot), value=0.0)
        return xpad[:, :, bot:bot + H, left:left + W]

    @staticmethod
    def _shift2d_roll(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Wrap-around shift (torus)."""
        return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

    def _shift(self, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        if self.zero_padded_shift:
            return self._shift2d_pad(x, dy, dx)
        return self._shift2d_roll(x, dy, dx)

    def forward(self, x: torch.Tensor, return_attention_map: bool = False):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Projections
        Q = self.query_proj(x)  # [B, d, H, W]
        K = self.key_proj(x)    # [B, d, H, W]
        M = self.msg_proj(x)    # [B, C, H, W]

        # Spatially pooled descriptors (global summaries)
        Q_pooled = Q.mean(dim=(2, 3))  # [B, d]
        # Precompute sender alive mask (dilated)
        if self.alive_to_alive:
            A_send = (F.max_pool2d(x[:, 3:4], 3, 1, 1) > self.alpha_thr).float()  # [B,1,H,W]

        # Randomly choose neighbor offsets this step
        k = min(self.num_neighbors, len(self.offsets))
        chosen = random.sample(self.offsets, k) if k > 0 else []

        messages = []
        logits   = []

        for (dy, dx) in chosen:
            K_shift = self._shift(K, dy, dx)  # [B, d, H, W]
            M_shift = self._shift(M, dy, dx)  # [B, C, H, W]

            # Alive→alive: only alive senders contribute
            if self.alive_to_alive:
                src = self._shift(A_send, dy, dx)  # [B,1,H,W]
                M_shift = M_shift * src

            # Logit uses pooled descriptors
            Kp = K_shift.mean(dim=(2, 3))            # [B, d]
            logit = (Q_pooled * Kp).sum(dim=1)       # [B]
            logits.append(logit)
            messages.append(M_shift)

        if len(chosen) == 0:
            # No neighbors — fall back to zeros
            agg_message = torch.zeros_like(M)
            if return_attention_map:
                attn_map = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
                return agg_message, attn_map
            return agg_message

        # Stack and softmax over offsets (numerically stable)
        L = torch.stack(logits, dim=0)                      # [N, B]
        L = L - L.max(dim=0, keepdim=True).values           # subtract max per batch item
        denom = self.scaling.abs() + 1e-6                   # positive temperature
        Wt = F.softmax(L / denom, dim=0)                    # [N, B]
        Wt = Wt.view(len(chosen), B, 1, 1, 1)               # broadcast to [N,B,1,1,1]

        M_stack = torch.stack(messages, dim=0)              # [N, B, C, H, W]
        weighted = M_stack * Wt                              # [N, B, C, H, W]
        agg_message = weighted.sum(dim=0)                   # [B, C, H, W]

        if return_attention_map:
            # Channel-averaged magnitude of weighted contributions, summed over offsets
            attn_map = weighted.abs().mean(dim=2).sum(dim=0)  # [B, H, W]
            # Normalize to [0,1] per-sample for visualization
            attn_min = attn_map.amin(dim=(1, 2), keepdim=True)
            attn_max = attn_map.amax(dim=(1, 2), keepdim=True)
            attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
            return agg_message, attn_map

        return agg_message
