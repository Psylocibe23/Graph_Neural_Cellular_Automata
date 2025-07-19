import json
from pathlib import Path


def load_config(config_path):
    """Load and validate the config.json file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
