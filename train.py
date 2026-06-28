"""
Unified training entry point.

Supports three modes controlled by config:

"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from config.schema import register_schemas
from utils.hardware import (
    set_gpu,
    release_trainer,
    stage_banner,
    suppress_teacher_wandb_in_sweep,
)

# Register structured-config schemas before Hydra composes (validates the
# `data` group against config/schema.py; yaml remains the source of values).
register_schemas()

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="train_sam")
def main(cfg: DictConfig):
    """Unified training entry point."""
    set_gpu(cfg)
    suppress_teacher_wandb_in_sweep(cfg)
    
    is_debug = cfg.get("debug", False)

    trainer = instantiate(cfg.trainer, cfg)
    trainer.setup(mode="train")

    if is_debug:
        log.info("[debug] Dry-run mode for teacher training.")
        if hasattr(trainer, "dry_run"):
            trainer.dry_run()
        else:
            log.info("[debug] Teacher trainer has no dry_run — skipping.")
    else:
        trainer.train()
        
    return

if __name__ == "__main__":
    main()
