import torch
import numpy as np


def count_parameters(model):
    """Count model's trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def damage_batch(batch, damage_size=10):
    """
    Damage images in batch to test regeneration abilities of the network
    damage_size is the side of the square of the damaged area in pixels    
    """
    B, C, H, W = batch.shape
    for b in range(B):
        y = np.random.randint(0, H - damage_size)
        x = np.random.randint(0, W - damage_size)
        batch[b, :, y:y+damage_size, x:x+damage_size]  = 0.0
    return batch