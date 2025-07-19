import torch
import random


class SamplePool:
    """
    A pool holding NCA states for persistence training
    - Allows sampling random batches
    - After each train step, updated states can replace originals
    - Can be added methods for perturbation/damage, shuffling, etc.
    """
    def __init__(self, pool_size, seed_fn, device="cpu"):
        """
        Args:
            pool_size (int): Total number of samples in the pool
            seed_fn (callable): Function returning a single seed state (torch.Tensor [C,H,W] or [B,C,H,W])
        """
        self.pool = [seed_fn(batch_size=1).squeeze(0).to(device) for _ in range(pool_size)]


    def sample(self, batch_size):
        """
        Sample a batch of indices and states
        Returns:
            idx (list[int]): List of indices in the pool
            batch (torch.Tensor): [B,C,H,W] batch of NCA states
        """
        idx = random.sample(range(len(self.pool)), batch_size)
        # Always clone so original pool isn't modified directly
        batch = torch.stack([self.pool[i].clone() for i in idx])
        return idx, batch

    def replace(self, idx, new_samples):
        """
        Replace samples in the pool with new NCA states after update

        Args:
            idx (list[int]): Indices to replace
            new_samples (torch.Tensor): [B,C,H,W] updated states
        """
        for i, s in zip(idx, new_samples):
            self.pool[i] = s.detach().clone()
