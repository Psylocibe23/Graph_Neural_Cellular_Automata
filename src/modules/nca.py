import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.perception import FixedSobelPerception


class NeuralCA(nn.Module):
    """
    Neural Cellular Automata with:
      - Fixed Sobel+identity perception (depthwise, frozen)
      - Two 1x1 convs, last conv zero-initialized
      - GroupNorm on dx (robust on sparse canvases / small batches)
      - Bounded update via tanh(dx) * update_gain
      - Pre-update gating (who is allowed to update)
      - Post-update gating ONLY on the alpha channel (preserve memory)
    """

    def __init__(
        self,
        n_channels: int,
        update_hidden: int = 128,
        img_size: int = 40,
        update_gain: float = 0.1,  # small step size keeps dynamics smooth
        alpha_thr: float = 0.1,  # alive threshold for masks
        use_groupnorm: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.img_size = img_size
        self.update_gain = update_gain
        self.alpha_thr = alpha_thr
        self.device = device

        # Fixed perception: depthwise conv with identity + Sobel (non-trainable)
        self.perception = FixedSobelPerception(n_channels)  # -> [B, 3*C, H, W]

        # Update network: 1x1 convs 
        in_dim = n_channels * 3
        self.update_net = nn.Sequential(
            nn.Conv2d(in_dim, update_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),  
            nn.Conv2d(update_hidden, n_channels, kernel_size=1, bias=False),
        )
        # Zero-init the last conv for gentle starts
        nn.init.zeros_(self.update_net[-1].weight)

        # Normalize the UPDATE field (dx), not the state x.
        # GroupNorm(1, C) ~ LayerNorm over channels
        self.norm = (
            nn.GroupNorm(1, n_channels, eps=1e-3, affine=True)
            if use_groupnorm else nn.Identity()
        )

    @torch.no_grad()
    def _alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alive mask from ALPHA channel, dilated by 3x3 max-pool.
        Returns [B,1,H,W] in {0,1}. 
        """
        alpha = x[:, 3:4]  # single channel (alpha)
        m = (F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > self.alpha_thr).float()
        return m

    def forward(self, x: torch.Tensor, fire_rate: float = 1.0) -> torch.Tensor:
        """
        One CA step:
          1) perception -> y
          2) update field -> dx
          3) optional stochastic firing
          4) pre-update alive gating on dx
          5) normalize + bound dx, then apply
          6) post-update gate ONLY alpha (no in-place ops)
        """
        # 1) Perception
        y = self.perception(x)  # [B, 3*C, H, W]

        # 2) Raw update field
        dx = self.update_net(y)  # [B, C, H, W]

        # 3) Optional stochastic firing (shared across channels)
        if fire_rate < 1.0:
            fire_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
            dx = dx * fire_mask

        # 4) Pre-update gating: only allow updates where current state is alive
        pre_alive = self._alive_mask(x)  # [B,1,H,W]
        dx = dx * pre_alive

        # 5) Normalize update, bound step size, and apply 
        dx = self.norm(dx)
        dx = torch.tanh(dx) * self.update_gain
        x = x + dx

        # 6) Post-update gating ONLY on alpha 
        post_alive = self._alive_mask(x)  # [B,1,H,W]
        if self.n_channels > 4:
            ones_pre  = torch.ones_like(x[:, :3])
            ones_post = torch.ones_like(x[:, 4:])
            gate = torch.cat([ones_pre, post_alive, ones_post], dim=1)
        else:
            ones_pre = torch.ones_like(x[:, :3])
            gate = torch.cat([ones_pre, post_alive], dim=1)
        x = x * gate 

        return x
