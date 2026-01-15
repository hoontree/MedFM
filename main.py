#!/usr/bin/env python3
"""
Main Entry Point for Multi-Model Training Framework

This script provides a unified interface for training and testing different models.

Usage:
    # Train with default config
    python main.py

    # Train specific model
    python main.py model=sam
    python main.py model=tinyusfm

    # Test mode
    python main.py mode=test model=sam checkpoint=/path/to/checkpoint.pth

    # Override parameters
    python main.py model=sam training.batch_size=64 training.base_lr=0.001

    # List available models
    python main.py --help
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from trainers import ModelBuilder


def setup_gpu(cfg: DictConfig):
    """Setup GPU environment."""
    if hasattr(cfg, 'hardware') and hasattr(cfg.hardware, 'gpu_ids'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.hardware.gpu_ids))

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Main training/testing function.

    Args:
        cfg: Configuration from Hydra
    """
    # Print available models if requested
    if cfg.get('list_models', False):
        models = ModelBuilder.list_models()
        print("Available Models:")
        for model in sorted(models):
            print(f"  - {model}")
        print("\nUsage:")
        print("  python main.py model=sam")
        print("  python main.py model=tinyusfm")
        print("  python main.py model=sam mode=test checkpoint=/path/to/checkpoint.pth")
        return

    # Validate configuration
    if not hasattr(cfg, 'model') or not hasattr(cfg.model, 'name'):
        print("Error: model.name is required in config")
        print("\nAvailable models:", ModelBuilder.list_models())
        print("\nUsage: python main.py model=sam")
        sys.exit(1)

    # Setup GPU
    setup_gpu(cfg)

    # Get mode
    mode = cfg.get('mode', 'train')

    # Print configuration
    print(f"Configuration (Mode: {mode}, Model: {cfg.model.name})")
    print(OmegaConf.to_yaml(cfg))
    # Create trainer
    print(f"Creating trainer for model: {cfg.model.name}")
    trainer = ModelBuilder.create_trainer(cfg)

    # Setup trainer
    trainer.setup(mode=mode)

    # Run training or testing
    if mode == 'train':
        print(f"Starting Training: {cfg.model.name}")

        # Check if test-only mode is enabled in config
        if cfg.get('test_only', {}).get('enabled', False):
            checkpoint_path = cfg.test_only.get('checkpoint_path')
            if checkpoint_path is None:
                checkpoint_path = cfg.get('checkpoint')

            if checkpoint_path is None:
                print("Error: checkpoint path is required for test-only mode")
                print("Use: test_only.checkpoint_path=/path/to/checkpoint.pth")
                print("Or: checkpoint=/path/to/checkpoint.pth")
                sys.exit(1)

            trainer.run_test_only(checkpoint_path)
        else:
            trainer.train()

    elif mode == 'test':
        print(f"Starting Testing: {cfg.model.name}")

        checkpoint_path = cfg.get('checkpoint')
        if checkpoint_path is None:
            checkpoint_path = cfg.get('test_only', {}).get('checkpoint_path')

        if checkpoint_path is None:
            print("Error: checkpoint is required for test mode")
            print("Use: python main.py mode=test checkpoint=/path/to/checkpoint.pth")
            sys.exit(1)

        trainer.run_test_only(checkpoint_path)

    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'test'")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
