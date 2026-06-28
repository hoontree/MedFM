import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from config.schema import register_schemas
from utils.hardware import set_gpu

# Register structured-config schemas before Hydra composes (validates the
# `data` group against config/schema.py; yaml remains the source of values).
register_schemas()


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
