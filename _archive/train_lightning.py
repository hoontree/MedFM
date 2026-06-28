"""
PyTorch Lightning training entry point.

Drop-in replacement for ``train.py`` that uses Lightning Trainer
instead of the custom BaseTrainer loop. All model / data / loss logic
is identical — only the orchestration layer changes.

Usage:
    python train_lightning.py                       # defaults (SAM)
    python train_lightning.py model=sam training.num_epochs=50
    python train_lightning.py training.batch_size=32
"""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from trainers.lightning_module import SAMLitModule
from trainers.lightning_datamodule import SegDataModule
from utils.hardware import set_gpu

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom callback: save LoRA-only checkpoint (matches original behaviour)
# ---------------------------------------------------------------------------

class LoRACheckpointCallback(L.Callback):
    """Save LoRA-only best / periodic checkpoints matching the
    original ``SAMTrainer._save_checkpoint`` semantics."""

    def __init__(self, ckpt_dir: Path, save_interval: int = 20):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.save_interval = save_interval
        self.best_dice = 0.0
        self.best_path: Path | None = None

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: SAMLitModule):
        dice = trainer.callback_metrics.get("val/Dice", 0.0)
        if isinstance(dice, torch.Tensor):
            dice = dice.item()

        epoch = trainer.current_epoch + 1

        # Best model
        if dice > self.best_dice:
            self.best_dice = dice
            path = self.ckpt_dir / f"best_epoch_{epoch}_dice{dice:.4f}.pth"
            pl_module.save_lora_parameters(str(path))
            self.best_path = path
            log.info("Saved best LoRA checkpoint: %s", path)

        # Periodic
        if epoch % self.save_interval == 0:
            path = self.ckpt_dir / f"epoch_{epoch}.pth"
            pl_module.save_lora_parameters(str(path))
            log.info("Saved periodic LoRA checkpoint: %s", path)


# ---------------------------------------------------------------------------
# Helper: build Lightning Trainer from Hydra config
# ---------------------------------------------------------------------------

def _build_trainer(cfg: DictConfig, exp_dir: Path) -> L.Trainer:
    """Construct a ``lightning.Trainer`` from the Hydra config,
    mirroring all features of the original training loop."""

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Callbacks ----
    callbacks = []

    # 1. Model checkpoint (Lightning-native, saves full state)
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-{val_Dice:.4f}",
            monitor="val/Dice",
            mode="max",
            save_top_k=1,
            verbose=True,
        )
    )

    # 2. LoRA-only checkpoint (original behaviour)
    save_interval = cfg.get("training", {}).get("save_interval", 20)
    callbacks.append(LoRACheckpointCallback(ckpt_dir, save_interval=save_interval))

    # 3. Early stopping
    es_cfg = cfg.get("training", {}).get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor="val/Dice",
                patience=es_cfg.get("patience", 15),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode="max",
                verbose=True,
            )
        )

    # 4. LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # ---- Loggers ----
    loggers = []

    # TensorBoard
    loggers.append(TensorBoardLogger(save_dir=str(exp_dir), name="tensorboard"))

    # WandB
    wandb_cfg = cfg.get("wandb", {})
    wandb_disabled = wandb_cfg.get("disabled", False)
    wandb_mode = wandb_cfg.get("mode", None)
    if wandb_disabled:
        wandb_mode = "disabled"

    loggers.append(
        WandbLogger(
            project=wandb_cfg.get("project", "medical_foundation_models"),
            entity=wandb_cfg.get("entity", "hheo"),
            save_dir=str(exp_dir),
            offline=(wandb_mode == "disabled"),
            log_model=wandb_cfg.get("log_model", False),
        )
    )

    # ---- Trainer kwargs ----
    num_epochs = cfg.training.get("num_epochs", 100)
    gpu_ids = cfg.get("hardware", {}).get("gpu_ids", [0])
    limit_train = cfg.training.get("limit_train_batches", None)

    # Gradient clipping (handled inside LightningModule.configure_gradient_clipping,
    # but we still set the flag so Lightning calls the hook)
    grad_clip = cfg.optimizer.get("gradient_clip", {})
    gradient_clip_val = (
        grad_clip.get("max_norm", 1.0) if grad_clip.get("enabled", False) else None
    )

    deterministic = cfg.get("hardware", {}).get("deterministic", True)
    # Use "warn" instead of True to avoid errors from ops without
    # deterministic CUDA implementations (e.g. upsample_bicubic2d_backward).
    if deterministic is True:
        deterministic = "warn"

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=loggers,
        deterministic=deterministic,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        limit_train_batches=limit_train if limit_train else 1.0,
        default_root_dir=str(exp_dir),
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm" if gradient_clip_val else None,
    )

    return trainer


# ---------------------------------------------------------------------------
# Experiment directory setup (mirrors BaseTrainer._setup_directories)
# ---------------------------------------------------------------------------

def _setup_exp_dir(cfg: DictConfig) -> Path:
    """Create the experiment directory tree."""
    from utils.distill_utils import get_experiment_tags

    model_name = cfg.model.get("sam_type", "") + "_" + cfg.model.name
    logs_root = Path(cfg.get("output", {}).get("dir", "logs"))
    phase = "train"

    exp_tags = get_experiment_tags(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = timestamp + ("_" + "_".join(exp_tags) if exp_tags else "")

    exp_dir = logs_root / phase / model_name / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_file = exp_dir / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)

    return exp_dir

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """PyTorch Lightning training entry point."""
    set_gpu(cfg)

    # Seed
    seed = cfg.get("hardware", {}).get("seed", 42)
    L.seed_everything(seed, workers=True)

    # Experiment directory
    exp_dir = _setup_exp_dir(cfg)
    log.info("Experiment directory: %s", exp_dir)

    # DataModule
    datamodule = SegDataModule(cfg)

    # LightningModule
    model = SAMLitModule(cfg)

    # Trainer
    trainer = _build_trainer(cfg, exp_dir)

    # ---- Train ----
    log.info("Starting Lightning training ...")
    trainer.fit(model, datamodule=datamodule)

    # ---- Test with best checkpoint ----
    log.info("Testing with best checkpoint ...")
    # Find best LoRA checkpoint
    lora_cb = next(
        (c for c in trainer.callbacks if isinstance(c, LoRACheckpointCallback)), None
    )
    if lora_cb and lora_cb.best_path and lora_cb.best_path.exists():
        model.load_lora_parameters(str(lora_cb.best_path))
        log.info("Loaded best LoRA checkpoint: %s", lora_cb.best_path)

    trainer.test(model, datamodule=datamodule)

    # Save test results alongside experiment directory
    test_metrics = trainer.callback_metrics
    results_path = exp_dir / "test_results.txt"
    with open(results_path, "w") as f:
        f.write("Test Results (Lightning)\n")
        if lora_cb and lora_cb.best_path:
            f.write(f"Best Model: {lora_cb.best_path.name}\n")
        for k, v in sorted(test_metrics.items()):
            if "final_test" in k:
                val = v.item() if hasattr(v, "item") else v
                f.write(f"{k}: {val:.4f}\n")
    log.info("Test results saved to %s", results_path)
    log.info("Training completed!")


if __name__ == "__main__":
    main()
