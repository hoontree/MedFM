"""
Model Builder Factory

This module provides a factory for creating trainers for different models.
"""

from hmac import new
from tabnanny import check
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from .base_trainer import BaseTrainer


class ModelBuilder:
    """Factory class for creating model trainers."""

    # Registry mapping model names to trainer module paths
    TRAINER_MAP = {
        "sam": ("trainers.sam_trainer", "SAMTrainer"),
        "sam_hybrid": ("trainers.sam_trainer", "SAMTrainer"),
        "vit_b": ("trainers.sam_trainer", "SAMTrainer"),
        "vit_l": ("trainers.sam_trainer", "SAMTrainer"),
        "vit_h": ("trainers.sam_trainer", "SAMTrainer"),
        "tinyusfm": ("trainers.tinyusfm_trainer", "TinyUSFMTrainer"),
        "usfm": ("trainers.tinyusfm_trainer", "TinyUSFMTrainer"),
        "segformer": ("trainers.segformer_trainer", "SegformerTrainer"),
        "sam3": ("trainers.sam3_adapter", "SAM3TrainerAdapter"),
        "ca_sam": ("trainers.ca_sam_trainer", "CASAMTrainer"),
        "casam": ("trainers.ca_sam_trainer", "CASAMTrainer"),
    }

    # Registry mapping model names to model creation functions or classes
    MODEL_MAP = {
        "sam": "model.sam_lora_image_encoder_mask_decoder",
        "vit_b": "model.sam_lora_image_encoder_mask_decoder",
        "vit_l": "model.sam_lora_image_encoder_mask_decoder",
        "vit_h": "model.sam_lora_image_encoder_mask_decoder",
        "tinyusfm": "model.tinyusfm_seg",
        "usfm": "model.usfm_seg",
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
    def create_model(
        cls, model_cfg: "DictConfig", num_classes: int, device: str = "cuda"
    ):
        """
        Create a raw model instance based on config.
        """
        model_name = model_cfg.name.lower()

        if "sam" in model_name or "vit" in model_name:
            from model.segment_anything import sam_model_registry
            from importlib import import_module

            sam_type = model_cfg.get("sam_type", model_name)
            if sam_type not in sam_model_registry:
                # Fallback for vit_b/l/h directly
                sam_type = (
                    "vit_b"
                    if "vit_b" in model_name
                    else ("vit_l" if "vit_l" in model_name else "vit_h")
                )

            sam, _ = sam_model_registry[sam_type](
                image_size=model_cfg.get("img_size", 224),
                num_classes=num_classes,
                checkpoint=model_cfg.get("sam_checkpoint", model_cfg.get("ckpt")),
                pixel_mean=[0, 0, 0],
                pixel_std=[1, 1, 1],
            )

            module_path = model_cfg.get(
                "module", "model.sam_lora_image_encoder_mask_decoder"
            )
            pkg = import_module(module_path)
            # Create a dict of all config items to pass as kwargs
            model_kwargs = {
                k: v for k, v in model_cfg.items() if k not in ["module", "name"]
            }
            model = pkg.LoRA_Sam(sam, model_cfg.get("rank", 4), **model_kwargs)

            lora_ckpt = model_cfg.get("lora_checkpoint", model_cfg.get("lora_ckpt"))
            if lora_ckpt:
                model.load_lora_parameters(lora_ckpt)

        elif "tinyusfm" in model_name:
            from model.tinyusfm_seg import SegmentationModel

            model = SegmentationModel(num_classes)
            try:
                checkpoint_path = model_cfg.get("checkpoint")
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    new_state_dict = {
                        k.replace("model.", "backbone."): v
                        for k, v in checkpoint.items()
                        if k.startswith("model.")
                    }
                    load_info = model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Failed to load checkpoint for tinyusfm model. Error: {e}")

        elif "usfm" in model_name:
            from model.usfm_seg import SegmentationModel

            model = SegmentationModel(num_classes)
            try:
                checkpoint_path = model_cfg.get("checkpoint")
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    state_dict = checkpoint
                    new_state_dict = {
                        f"transformer.{k}": v for k, v in state_dict.items()
                    }
                    load_info = model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Failed to load checkpoint for usfm model. Error: {e}")
        else:
            raise ValueError(f"Unsupported model for raw creation: {model_name}")
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {model_name}")
        print(f"Total parameters: {total_params:,}")
        return model

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
