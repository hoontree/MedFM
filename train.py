"""
Unified training entry point.

Supports three modes controlled by config:

  1. Teacher training only (default):
       python train.py model=sam

  2. Distillation only:
       python distill.py  (separate entry — uses config/distill.yaml)

  3. Teacher training → distillation pipeline:
       python train.py pipeline.enabled=true model=sam
"""

import logging
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Distillation runner
# ---------------------------------------------------------------------------

def _run_distillation(cfg: DictConfig, teacher_trainer) -> Optional[dict]:
    """Run the distillation stage after teacher training.

    Returns a result dict with ``final_metrics`` and ``log_dir``.
    Raises on any error.
    """
    if not cfg.get("pipeline", {}).get("distill", {}).get("enabled", True):
        log.info("Distillation stage disabled — skipping.")
        return None

    # 1. Get teacher checkpoint and pipeline context
    teacher_ckpt = Path(teacher_trainer.best_model_path)
    teacher_run_id = getattr(teacher_trainer, "wandb_run_id", None)
    teacher_log_dir = (
        str(teacher_trainer.exp_dir)
        if getattr(teacher_trainer, "exp_dir", None)
        else None
    )

    # 2. Free teacher GPU memory
    release_trainer(teacher_trainer)

    # 3. Build distillation config
    distill_cfg = build_distill_cfg(
        cfg,
        teacher_ckpt,
        teacher_run_id=teacher_run_id,
        teacher_log_dir=teacher_log_dir,
    )

    stage_banner("STAGE 2: Knowledge Distillation")
    log.info("Teacher checkpoint: %s", teacher_ckpt.resolve())

    # 4. Run distillation
    distill_trainer = instantiate(distill_cfg.trainer, distill_cfg)
    return distill_trainer.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Unified training entry point with optional pipeline distillation."""
    set_gpu(cfg)
    suppress_teacher_wandb_in_sweep(cfg)

    is_pipeline = cfg.get("pipeline", {}).get("enabled", False)
    is_debug = cfg.get("debug", False)

    # --- Stage 1: Teacher training ---
    if is_pipeline:
        stage_banner("STAGE 1: Teacher Training")

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

    if not is_pipeline:
        return

    # --- Stage 2: Pipeline distillation ---
    result = _run_distillation(cfg, trainer)
    if not result:
        return

    # --- Report ---
    stage_banner("PIPELINE SUMMARY")
    final_metrics = result.get("final_metrics", {})
    for k, v in final_metrics.items():
        if isinstance(v, (float, int)):
            log.info("  %s = %.6f", k, v)

    if result.get("log_dir"):
        log.info("Distillation logs: %s", result["log_dir"])


if __name__ == "__main__":
    main()
