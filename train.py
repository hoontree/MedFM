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
    set_gpu(cfg)  # sets CUDA_VISIBLE_DEVICES only; no CUDA context yet
    suppress_teacher_wandb_in_sweep(cfg)

    # SAM3 native multi-GPU (DDP) training runs through SAM3's own Trainer,
    # which spawns one process per GPU and initializes torch.distributed. That
    # must happen from a process with no CUDA context yet, so route here —
    # before `instantiate(cfg.trainer, ...)` below creates the model on a GPU.
    # (Metrics for this path are computed afterwards by loading the saved
    # checkpoint into the single-GPU SAM3TrainerAdapter — the native trainer's
    # own eval is detection-style, not this project's pixel Dice suite.)
    if cfg.get("sam3", {}).get("use_native_trainer", False):
        from trainers.sam3_adapter import run_sam3_orchestrator
        log.info("SAM3 native trainer enabled — launching multi-GPU orchestrator.")
        run_sam3_orchestrator(cfg)
        return

    test_only_cfg = cfg.get("test_only", {})
    if test_only_cfg.get("enabled", False):
        checkpoint_path = test_only_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("test_only.enabled=true requires test_only.checkpoint_path=<path>")
        log.info(f"TEST-ONLY: loading {checkpoint_path} — skipping training entirely.")
        trainer = instantiate(cfg.trainer, cfg)
        trainer.setup(mode="test")
        trainer.run_test_only(checkpoint_path)
        return

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
