# modules/nca_graph.py
# -----------------------------------------------------------------------------
# Graph-augmented Neural Cellular Automata (stable long-horizon)
#
# Key ideas:
#   • Classic NCA core (fixed Sobel perception → 1×1 MLP → bounded update)
#   • GroupNorm on UPDATE (dx), not on the state (x)
#   • Alive gating: pre-update on dx; post-update on ALPHA ONLY
#   • Mid-range GraphAugmentation is injected as a bounded residual into dx
#     (optionally only into hidden channels), then normalized/bounded together
#     with the local update so it can't destabilize dynamics.
#
# Usage:
#   from modules.nca_graph import NeuralCAGraph
#   model = NeuralCAGraph(...)
#   x = model(x, fire_rate=0.5)                      # classic inference
#   x, attn = model(x, fire_rate=0.5, return_attention=True)  # with viz
# -----------------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.perception import FixedSobelPerception
from modules.graph_augmentation import GraphAugmentation


class NeuralCAGraph(nn.Module):
    """
    Neural Cellular Automata with graph-gated mid-range message passing.

    Components:
      - Perception: fixed depthwise conv (identity + Sobel) → [B, 3*C, H, W]
      - Update MLP: 1×1 conv → ReLU (NOT in-place) → 1×1 conv (last layer zero-init)
      - GroupNorm on dx, not on x
      - Bounded step: dx = tanh(dx) * update_gain
      - Alive gating: pre-update on dx; post-update on alpha channel only
      - GraphAugmentation: produces a per-pixel message [B, C, H, W]
        • message is bounded via tanh and scaled by message_gain
        • optionally zeroed on RGB+alpha (hidden_only=True)
        • added into dx BEFORE norm/bound so it’s treated like any update

    Args:
        n_channels:      total channels (RGBA + hidden)
        update_hidden:   hidden width of the update MLP
        img_size:        canvas size (for LayerNorm alternatives; unused directly)
        update_gain:     step size after bounding tanh(dx)
        alpha_thr:       alive threshold for alpha (post 3×3 max-pool)
        use_groupnorm:   if True, apply GroupNorm(1, C) to dx
        # Graph options:
        message_gain:        scale factor for bounded graph message
        hidden_only:         if True, graph message only affects channels 4:
        graph_d_model:       Q/K proj dim
        graph_attention_radius:  maximum |dx|,|dy| for mid-range offsets (excl. 3×3 local)
        graph_num_neighbors: number of neighbor offsets sampled each step
        graph_gating_hidden: hidden width for channel-wise gate MLP
        device:          device string
    """

    def __init__(
        self,
        n_channels: int,
        update_hidden: int = 128,
        img_size: int = 40,
        update_gain: float = 0.1,
        alpha_thr: float = 0.1,
        use_groupnorm: bool = True,
        *,
        message_gain: float = 0.5,
        hidden_only: bool = True,
        graph_d_model: int = 16,
        graph_attention_radius: int = 4,
        graph_num_neighbors: int = 8,
        graph_gating_hidden: int = 32,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.img_size = img_size
        self.update_gain = float(update_gain)
        self.alpha_thr = float(alpha_thr)
        self.device = device

        # --- Perception (frozen depthwise: identity + Sobel) ---
        self.perception = FixedSobelPerception(n_channels)  # -> [B, 3*C, H, W]

        # --- Local update MLP (1×1 convs, zero-init last layer) ---
        in_dim = n_channels * 3
        self.update_net = nn.Sequential(
            nn.Conv2d(in_dim, update_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),  # NEVER in-place (autograd safety)
            nn.Conv2d(update_hidden, n_channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

        # --- Normalize UPDATE (dx), not the state ---
        self.norm = nn.GroupNorm(1, n_channels, eps=1e-3, affine=True) if use_groupnorm else nn.Identity()

        # --- Graph augmentation module ---
        self.graph = GraphAugmentation(
            n_channels=n_channels,
            d_model=graph_d_model,
            attention_radius=graph_attention_radius,
            num_neighbors=graph_num_neighbors,
            gating_hidden=graph_gating_hidden,
        )

        # How we inject the graph message
        self.message_gain = float(message_gain)
        self.hidden_only = bool(hidden_only)

    @torch.no_grad()
    def _alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alive mask from ALPHA channel, dilated by 3×3 max-pool.
        Returns [B,1,H,W] (float in {0,1}); no gradients needed.
        """
        alpha = x[:, 3:4]
        m = (F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > self.alpha_thr).float()
        return m

    def _apply_message_policy(self, m: torch.Tensor) -> torch.Tensor:
        """
        Apply channel policy (hidden_only) and magnitude bounding to graph message.
        """
        if self.hidden_only and m.shape[1] >= 4:
            # zero-out RGB+alpha, keep hidden channels only
            zeros4 = torch.zeros_like(m[:, :4])
            m = torch.cat([zeros4, m[:, 4:]], dim=1)
        # bound and scale message before adding to dx
        m = torch.tanh(m) * self.message_gain
        return m

    def forward(
        self,
        x: torch.Tensor,
        fire_rate: float = 1.0,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        One CA step with mid-range graph message.

        Pipeline:
          1) y = perception(x)
          2) dx_local = update_net(y)
          3) m = graph(x)         (optionally return attention map)
          4) dx = dx_local + policy(m)   (bounded/scaled, hidden-only if set)
          5) fire-rate mask on dx (stochastic updates)
          6) pre-alive gate on dx
          7) dx = norm(dx); dx = tanh(dx) * update_gain
          8) x = x + dx
          9) post-alive gate on ALPHA ONLY (out-of-place)
        """
        # 1) Perception
        y = self.perception(x)  # [B, 3*C, H, W]

        # 2) Local update field
        dx = self.update_net(y)  # [B, C, H, W]

        # 3) Graph message (mid-range)
        if return_attention:
            m, attn_map = self.graph(x, return_attention_map=True)   # m: [B,C,H,W], attn_map: [B,H,W]
        else:
            m = self.graph(x)                                        # m: [B,C,H,W]
            attn_map = None

        # Merge message into dx via bounded residual (policy may zero-out RGB+alpha)
        dx = dx + self._apply_message_policy(m)

        # 5) Optional stochastic firing mask (shared across channels)
        if fire_rate < 1.0:
            fire_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
            dx = dx * fire_mask

        # 6) Pre-update alive gating: updates only where x is alive (by alpha)
        pre_alive = self._alive_mask(x)  # [B,1,H,W]
        dx = dx * pre_alive

        # 7) Normalize + bound update, then apply (out-of-place add)
        dx = self.norm(dx)
        dx = torch.tanh(dx) * self.update_gain
        x = x + dx

        # 9) Post-update gate ONLY on alpha channel (no in-place slicing)
        post_alive = self._alive_mask(x)  # [B,1,H,W]
        if self.n_channels > 4:
            ones_pre  = torch.ones_like(x[:, :3])
            ones_post = torch.ones_like(x[:, 4:])
            gate = torch.cat([ones_pre, post_alive, ones_post], dim=1)
        else:
            ones_pre = torch.ones_like(x[:, :3])
            gate = torch.cat([ones_pre, post_alive], dim=1)
        x = x * gate

        return (x, attn_map) if return_attention else x
