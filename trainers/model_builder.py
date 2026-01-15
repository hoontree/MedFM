"""
Model Builder Factory

This module provides a factory for creating trainers for different models.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from .base_trainer import BaseTrainer


class ModelBuilder:
    """Factory class for creating model trainers."""

    # Registry mapping model names to trainer module paths
    TRAINER_MAP = {
        'sam': ('trainers.sam_trainer', 'SAMTrainer'),
        'vit_b': ('trainers.sam_trainer', 'SAMTrainer'),
        'vit_l': ('trainers.sam_trainer', 'SAMTrainer'),
        'vit_h': ('trainers.sam_trainer', 'SAMTrainer'),
        'tinyusfm': ('trainers.tinyusfm_trainer', 'TinyUSFMTrainer'),
        'usfm': ('trainers.tinyusfm_trainer', 'TinyUSFMTrainer'),
        'segformer': ('trainers.segformer_trainer', 'SegformerTrainer'),
    }

    @classmethod
    def create_trainer(cls, cfg: "DictConfig") -> "BaseTrainer":
        """
        Create a trainer based on the model name in config.

        Args:
            cfg: Configuration object

        Returns:
            Trainer instance

        Raises:
            ValueError: If model name is not supported
        """
        import importlib

        model_name = cfg.model.name.lower()

        if model_name not in cls.TRAINER_MAP:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {list(cls.TRAINER_MAP.keys())}"
            )

        module_path, class_name = cls.TRAINER_MAP[model_name]
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        return trainer_class(cfg)

    @classmethod
    def register_trainer(cls, name: str, module_path: str, class_name: str):
        """
        Register a new trainer class.

        Args:
            name: Model name
            module_path: Module path (e.g., 'trainers.sam_trainer')
            class_name: Class name (e.g., 'SAMTrainer')
        """
        cls.TRAINER_MAP[name.lower()] = (module_path, class_name)

    @classmethod
    def list_models(cls) -> list:
        """Get list of available models."""
        return list(cls.TRAINER_MAP.keys())
