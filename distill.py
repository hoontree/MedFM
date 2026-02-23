import hydra
from omegaconf import DictConfig
import os


@hydra.main(version_base=None, config_path="config", config_name="distill")
def main(cfg: DictConfig):
    gpu_ids = cfg.get("hardware", {}).get("gpu_ids", cfg.get("gpu_ids", [0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    from trainers.model_builder import ModelBuilder

    trainer = ModelBuilder.create_trainer(cfg)

    if cfg.get("debug", False):
        trainer.dry_run()
        return

    trainer.train()


if __name__ == "__main__":
    main()
