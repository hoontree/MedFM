"""Hardware, GPU, and runtime environment utilities."""

import gc
import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def set_gpu(cfg: DictConfig) -> None:
    """Set CUDA_VISIBLE_DEVICES from config."""
    gpu_ids = cfg.get("hardware", {}).get("gpu_ids", [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def free_gpu() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_trainer(trainer) -> None:
    """Release all trainer GPU resources and free memory."""
    for attr in ("model", "optimizer", "scheduler",
                 "train_loader", "val_loader", "test_loader",
                 "teacher", "student", "distiller"):
        obj = getattr(trainer, attr, None)
        if obj is None:
            continue
        if hasattr(obj, "to"):
            try:
                obj.to("cpu")
            except Exception:
                pass
        setattr(trainer, attr, None)
    free_gpu()


def suppress_teacher_wandb_in_sweep(cfg: DictConfig) -> None:
    """Disable teacher W&B logging during pipeline sweep runs."""
    if not cfg.get("pipeline", {}).get("enabled", False):
        return
    if os.environ.get("WANDB_SWEEP_ID") is None:
        return
    OmegaConf.set_struct(cfg, False)
    if "wandb" not in cfg or cfg.wandb is None:
        cfg.wandb = OmegaConf.create({})
    cfg.wandb.disabled = True
    log.info("Sweep detected — teacher W&B disabled.")


def stage_banner(title: str) -> None:
    """Print a visible stage separator."""
    sep = "=" * 50
    log.info("\n%s\n  %s\n%s", sep, title, sep)
