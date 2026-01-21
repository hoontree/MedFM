import os
import hydra
from omegaconf import DictConfig, OmegaConf

from trainers.model_builder import ModelBuilder


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Main training entry point using unified Trainer system."""
    # Set GPU environment
    gpu_ids = cfg.get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # Print config for debugging
    print(OmegaConf.to_yaml(cfg))

    # Create trainer based on model type
    trainer = ModelBuilder.create_trainer(cfg)

    # Setup and run training
    trainer.setup(mode="train")
    trainer.train()


if __name__ == "__main__":
    main()
