import yaml
import os
from datetime import datetime
import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_seed(seed):
    """
    Set the seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

# Utility functions
def load_yaml(file_path):
    """Load a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(*configs):
    """Merge multiple YAML configuration dictionaries."""
    merged = {}
    for config in configs:
        merged.update(config)
    return merged


def merge_args(base_config, cli_args):
    """
    Merge the base configuration with command-line arguments.
    Command-line arguments will override the base configuration.
    """
    for key, value in vars(cli_args).items():
        if value is not None:
            # Update nested dictionaries
            keys = key.split('.')
            target = base_config
            for subkey in keys[:-1]:
                target = target.get(subkey, {})
            target[keys[-1]] = value
    return base_config


def save_config(config, output_path):
    """Save a YAML configuration to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        yaml.dump(config, file)
    print(f"Configuration saved to {output_path}")


def create_experiment_dir(config, seed, base_dir="results"):
    """
    Create a hierarchical directory structure for the experiment based on the configuration.
    """
    # Extract key components for the directory hierarchy
    dataset_name = config['data']['dataset_name']
    if dataset_name == 'imagenet':
        dataset_name += '_' + config['data']['d2_name']
    task = config['model'].get('task', 'classification')  # e.g., regression or classification
    extractor_type = config['model']['feature_extractor_type']

    # Construct hierarchical path
    experiment_dir = os.path.join(
        base_dir,
        dataset_name,        # Top-level: Dataset name
        task,                # Second level: Task type (classification or regression)
        extractor_type,      # Third level: Feature extractor type
        str(seed)
    )

    # Create the directory hierarchy
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the configuration for reproducibility
    config_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_path)  # Assuming save_config is implemented to save YAML files

    return experiment_dir


def get_config_value(config, key):
    value = config.get(key, None)
    return float(value) if value is not None else None