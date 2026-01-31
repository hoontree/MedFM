import hydra
from omegaconf import DictConfig
from trainers.distill_trainer import DistillTrainer
import os


@hydra.main(version_base=None, config_path="config", config_name="distill")
def main(cfg: DictConfig):
    gpu_ids = cfg.get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    trainer = DistillTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
