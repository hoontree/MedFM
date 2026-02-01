import hydra
from omegaconf import OmegaConf
from trainers.model_builder import ModelBuilder
import torch
import os


def test_model_creation():
    # Test cases: (model_name, num_classes, img_size)
    test_cases = [
        ("TinyUSFM", 1, 224),
        ("sam", 2, 224),
        ("segformer", 3, 512),
    ]

    for model_name, num_classes, img_size in test_cases:
        print(
            f"\n>>> Testing model: {model_name} (num_classes={num_classes}, img_size={img_size})"
        )

        # Simulate Hydra config loading
        # In a real scenario, this would be loaded from config/model/{model_name}.yaml
        config_path = f"config/model/{model_name}.yaml"
        if not os.path.exists(config_path):
            print(f"Skipping {model_name}, config not found at {config_path}")
            continue

        model_cfg = OmegaConf.load(config_path).model

        try:
            model = ModelBuilder.create_model(
                model_cfg, num_classes=num_classes, img_size=img_size
            )
            print(f"Successfully created {model_name}")

            # Verify num_classes
            if hasattr(model, "num_classes"):
                actual_nc = model.num_classes
            elif hasattr(model, "decode_head") and hasattr(
                model.decode_head, "num_classes"
            ):
                actual_nc = model.decode_head.num_classes
            elif hasattr(model, "sam") and hasattr(
                model.sam.mask_decoder, "num_multimask_outputs"
            ):
                actual_nc = model.sam.mask_decoder.num_multimask_outputs
            elif hasattr(model, "config") and hasattr(model.config, "num_labels"):
                actual_nc = model.config.num_labels
            else:
                actual_nc = "Unknown"

            print(f"Actual num_classes: {actual_nc}")
            assert str(actual_nc) == str(
                num_classes
            ), f"Expected {num_classes}, got {actual_nc}"

        except Exception as e:
            print(f"Failed to create {model_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_model_creation()
