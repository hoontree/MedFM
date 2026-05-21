"""
Unified training entry point.

Supports three modes controlled by config:

"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.hardware import (
    set_gpu,
    release_trainer,
    stage_banner,
    suppress_teacher_wandb_in_sweep,
)
from utils.pipeline import build_distill_cfg

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="train_sam")
def main(cfg: DictConfig):
    """Unified training entry point with optional pipeline distillation."""
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
