import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.hardware import set_gpu


@hydra.main(version_base=None, config_path="config", config_name="distill_sam_to_usfm_binary")
def main(cfg: DictConfig):
    set_gpu(cfg)

    trainer = instantiate(cfg.trainer, cfg)

    if cfg.get("debug", False):
        trainer.dry_run()
        return

    trainer.train()


if __name__ == "__main__":
    main()
