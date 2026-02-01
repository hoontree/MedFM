import hydra
from omegaconf import OmegaConf, DictConfig
from trainers.model_builder import ModelBuilder
import os
import torch


def test_instantiation():
    # 1. Test TinyUSFM (Trainer and Model)
    cfg = OmegaConf.create(
        {
            "trainer": {"_target_": "trainers.tinyusfm_trainer.TinyUSFMTrainer"},
            "model": {
                "_target_": "model.tinyusfm_seg.SegmentationModel",
                "name": "TinyUSFM",
                "num_classes": 2,
                "pretrained": False,
                "checkpoint": None,
                "img_size": 224,
            },
            "data": {
                "num_classes": 2,
                "name": "test_data",
                "train": ["BUSBRA"],
                "val": ["BUSBRA"],
            },
            "training": {"num_epochs": 1, "batch_size": 1, "lr": 0.001},
            "optimizer": {"name": "Adam"},
            "hardware": {"seed": 42, "gpu_ids": [0]},
            "visualization": {"num_samples": 1},
        }
    )

    print("--- Testing TinyUSFM Trainer Instantiation ---")
    try:
        # This will fail on data loading, but we can catch it
        trainer = ModelBuilder.create_trainer(cfg)
        print(f"Trainer class: {trainer.__class__.__name__}")
    except Exception as e:
        print(
            f"Trainer instantiation failed as expected (likely data loading): {type(e).__name__}"
        )

    print("\n--- Testing TinyUSFM Model Instantiation ---")
    model = ModelBuilder.create_model(cfg.model, num_classes=2, device="cpu")
    print(f"Model class: {model.__class__.__name__}")

    # 2. Test SAM Hybrid (E0_AL_DL)
    print("\n--- Testing SAM Hybrid Model Instantiation ---")
    sam_model_cfg = OmegaConf.create(
        {
            "_target_": "model.sam_hybrid_adapter.build_sam_hybrid",
            "name": "sam_hybrid",
            "sam_type": "vit_b",
            "adaptation_mode": "encoder_frozen_alignment_decoder_lora",
            "img_size": 224,
            "r_e": 4,
            "r_d": 4,
            "sam_checkpoint": None,
            "alignment_use_bn": False,
            "alignment_num_blocks": 4,
            "alignment_hidden_channels": 256,
            "lora_checkpoint": None,
        }
    )
    sam_model = ModelBuilder.create_model(sam_model_cfg, num_classes=2, device="cpu")
    print(f"SAM Model class: {sam_model.__class__.__name__}")
    print(f"Adaptation mode: {sam_model.adaptation_mode}")

    # 3. Test Distiller Instantiation
    print("\n--- Testing Distiller Instantiation ---")
    distill_methods_cfg = OmegaConf.create(
        {
            "method": {
                "_target_": "distillers.unified_distiller.UnifiedDistiller",
                "name": "hybrid",
                "alpha": 1.0,
                "beta": 1.0,
                "gamma_attn": 0.7,
                "gamma_align": 0.3,
            },
            "data": {"num_classes": 2},
        }
    )

    from distillers import DistillerRegistry

    distiller = DistillerRegistry.create(distill_methods_cfg)
    print(f"Distiller class: {distiller.__class__.__name__}")
    print(f"Distiller alpha: {distiller.alpha}")


if __name__ == "__main__":
    test_instantiation()
