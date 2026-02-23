"""
Model Builder Factory

This module provides a factory for creating trainers and models using Hydra instantiate.
All models should define _target_ in their config for Hydra instantiation.
Checkpoint loading is handled automatically via the model's load_checkpoint method.
"""

from typing import TYPE_CHECKING
from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from .base_trainer import BaseTrainer


class ModelBuilder:
    """Factory class for creating model trainers and models via Hydra."""

    @classmethod
    def create_trainer(cls, cfg: "DictConfig") -> "BaseTrainer":
        """Create a trainer via Hydra instantiate."""
        print(f"Instantiating trainer: {cfg.trainer._target_}")
        return instantiate(cfg.trainer, cfg)

    @classmethod
    def create_model(cls, model_cfg: "DictConfig", **kwargs):
        """
        Create a model instance via Hydra instantiate.

        The model config must contain _target_ pointing to a model class or factory function.
        If the model has a load_checkpoint method and checkpoint is in config, it will be called.
        Supports 'checkpoint=auto' to automatically find the best checkpoint based on hyperparameters.

        Args:
            model_cfg: Hydra config with _target_ and model parameters
            **kwargs: Additional parameters (num_classes, img_size) to inject if missing

        Returns:
            Instantiated model with checkpoint loaded if applicable
        """
        # Inject missing parameters from kwargs
        if "num_classes" not in model_cfg and "num_classes" in kwargs:
            model_cfg.num_classes = kwargs["num_classes"]
        if "img_size" not in model_cfg and "img_size" in kwargs:
            model_cfg.img_size = kwargs["img_size"]

        if "_target_" not in model_cfg:
            raise ValueError(f"Model config must have _target_. Got: {dict(model_cfg)}")

        print(f"Instantiating model: {model_cfg._target_}")
        model = instantiate(model_cfg)

        # Auto-load checkpoint if model supports it
        checkpoint_path = model_cfg.get("checkpoint")

        # Resolve 'auto' checkpoint path
        if checkpoint_path == "auto":
            from utils.distill_utils import find_best_checkpoint

            resolved_path = find_best_checkpoint(model_cfg)
            if resolved_path:
                print(f"Resolved 'auto' checkpoint to: {resolved_path}")
                checkpoint_path = str(resolved_path)
            else:
                msg = f"Error: Could not resolve 'auto' checkpoint for {model_cfg.get('name', 'model')}. Ensure the teacher model has been trained and logs are available in 'logs/'."
                print(msg)
                raise FileNotFoundError(msg)

        if checkpoint_path and hasattr(model, "load_checkpoint"):
            model.load_checkpoint(checkpoint_path)

        # Log model info
        model_name = model_cfg.get("name", model_cfg._target_.split(".")[-1])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name} | Parameters: {total_params:,}")

        return model
