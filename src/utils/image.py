import os
from PIL import Image
import torch
import numpy as np


def load_rgba_image(path, size):
    """Load an image as RGBA, resize, normalize, and return as torch tensor [C, H, W]"""
    img = Image.open(path).convert("RGBA").resize((size, size), Image.BILINEAR)
    img = np.array(img) / 255.0   # Scale to [0,1]
    img = torch.from_numpy(img).float().permute(2, 0, 1)  # [C, H, W]
    return img


def load_single_target_image(config):
    """
    Loads only the image specified in config["data"]["active_target"]
    Returns a torch tensor [C, H, W]
    """
    emoji_dir = config["data"]["emojis_dir"]
    img_size = config["data"]["img_size"]
    target_name = config["data"]["active_target"]

    if "targets" in config["data"]:
        assert target_name in config["data"]["targets"], (
            f"{target_name} is not listed in config['data']['targets']!"
        )

    path = os.path.join(emoji_dir, target_name)
    assert os.path.isfile(path), f"Image {path} not found!"

    return load_rgba_image(path, img_size)
