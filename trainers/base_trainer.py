"""
Base Trainer for Multi-Model Training Framework

This module provides a base class for training different models with common functionalities.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import wandb

from utils.logger import setup_logger
from utils.evaluate import Evaluator_seg


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like Dice (higher is better), 'min' for loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if score improved."""
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.early_stop


class BaseTrainer(ABC):
    """
    Base trainer class that provides common training infrastructure.

    All model-specific trainers should inherit from this class and implement
    the abstract methods for model creation, data loading, and training logic.
    """

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None):
        """
        Initialize base trainer.

        Args:
            cfg: Configuration object (OmegaConf DictConfig)
            model: Pre-built model instance (optional)
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize attributes
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.logger = None
        self.evaluator = Evaluator_seg()
        self.early_stopping = None

        # Directories
        self.exp_dir = None
        self.ckpt_dir = None
        self.log_dir = None

        # Training state
        self.best_metric = 0.0
        self.best_model_path = None
        self.current_epoch = 0
        self.global_step = 0

    def setup(self, mode: str = "train"):
        """
        Setup training environment.

        Args:
            mode: 'train' or 'test'
        """
        # Set random seeds
        self._set_seed()

        # Setup directories
        self._setup_directories(mode)

        # Setup logger
        self._setup_logger()

        # Setup wandb
        if mode == "train":
            self._setup_wandb()

        # Create data loaders
        self._create_dataloaders()

        # Create model
        self._create_model()

        # Setup training components (only for training mode)
        if mode == "train":
            self._create_optimizer()
            self._create_scheduler()
            self._setup_early_stopping()

        self.logger.info(f"Setup completed for {mode} mode")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.cfg.get("hardware", {}).get("seed", 1234)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        deterministic = self.cfg.get("hardware", {}).get("deterministic", True)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def _setup_directories(self, mode: str):
        """Setup experiment directories."""
        # Get model name
        model_name = self.cfg.model.get("sam_type", "") + "_" + self.cfg.model.name

        # Create base directory
        logs_root = Path(self.cfg.get("output", {}).get("dir", "logs"))

        # Determine if this is an adaptation phase (pre-distillation training)
        distill_cfg = self.cfg.get("distillation", {})
        phase = "train"
        if (
            distill_cfg.get("enabled", False)
            and distill_cfg.get("phase") == "adaptation"
        ):
            phase = "adaptation"

        # Create experiment tags
        exp_tags = self._create_exp_tags()

        # Create timestamp-based experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir_name = timestamp + ("_" + "_".join(exp_tags) if exp_tags else "")

        # Final experiment directory
        self.exp_dir = logs_root / phase / model_name / exp_dir_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir = self.exp_dir

        # Save config
        config_file = self.exp_dir / "config.yaml"
        with open(config_file, "w") as f:
            OmegaConf.save(self.cfg, f)

    def _create_exp_tags(self) -> list:
        """Create experiment tags based on hyperparameters."""
        from utils.distill_utils import get_experiment_tags

        return get_experiment_tags(self.cfg)

    def _setup_logger(self):
        """Setup logger."""
        log_file = self.exp_dir / "train.log"
        self.logger = setup_logger(str(log_file), logger_name="medfm.train")

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb_config = self.cfg.get("wandb", {})
        wandb_mode = wandb_config.get("mode", None)
        if wandb_config.get("disabled", False):
            wandb_mode = "disabled"

        # In a sweep, wandb handles the run name. For regular runs, we use a custom name.
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        run_name = (
            None
            if is_sweep
            else f"{self.cfg.model.get('encoder_mode', 'default')}_encoder_{self.cfg.model.get('decoder_mode', 'default')}_decoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb_run_config = self._build_wandb_config(is_sweep=is_sweep)

        self.wandb_run = wandb.init(
            entity=wandb_config.get("entity", "hheo"),
            project=wandb_config.get("project", "medical_foundation_models"),
            name=run_name,
            config=wandb_run_config,
            dir=str(self.exp_dir),
            mode=wandb_mode,
        )

    def _build_wandb_config(self, is_sweep: bool) -> Dict[str, Any]:
        """Build W&B config payload while avoiding sweep key-type collisions."""
        wandb_run_config = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(wandb_run_config, dict):
            return {}

        if is_sweep and isinstance(wandb_run_config.get("model"), dict):
            wandb_run_config["model_cfg"] = wandb_run_config.pop("model")

        return wandb_run_config

    def _setup_early_stopping(self):
        """Setup early stopping."""
        early_stop_cfg = self.cfg.get("training", {}).get("early_stopping", {})

        if early_stop_cfg.get("enabled", False):
            patience = early_stop_cfg.get("patience", 15)
            min_delta = early_stop_cfg.get("min_delta", 0.001)
            self.early_stopping = EarlyStopping(
                patience=patience, min_delta=min_delta, mode="max"
            )
            self.logger.info(
                f"Early stopping enabled: patience={patience}, min_delta={min_delta}"
            )

    @abstractmethod
    def _create_model(self):
        """Create and initialize model. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _create_dataloaders(self):
        """Create train/val/test data loaders. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _create_optimizer(self):
        """Create optimizer. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _create_scheduler(self):
        """Create learning rate scheduler. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        raise NotImplementedError

    @abstractmethod
    def test(self) -> Dict[str, float]:
        """
        Test model.

        Returns:
            Dictionary of test metrics
        """
        raise NotImplementedError

    def _iter_test_loaders(self):
        """Iterate over test loaders yielding (name, loader) tuples."""
        if isinstance(self.test_loader, dict):
            yield from self.test_loader.items()
        else:
            yield "test", self.test_loader

    def _log_data_sizes(self):
        """Log dataset sizes. Call from _create_dataloaders()."""
        self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")
        if isinstance(self.test_loader, dict):
            total = sum(len(l.dataset) for l in self.test_loader.values())
            self.logger.info(f"Test set size (Total): {total}")
        for name, loader in self._iter_test_loaders():
            prefix = "  - " if isinstance(self.test_loader, dict) else ""
            self.logger.info(f"{prefix}{name}: {len(loader.dataset)}")

    def _log_model_info(self):
        """Log total and trainable parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total:,}")
        self.logger.info(f"Trainable parameters: {trainable:,}")
        if wandb.run is not None:
            wandb.summary["model/total_params"] = total
            wandb.summary["model/trainable_params"] = trainable

    def _log_step_metrics(self, metrics: Dict, step: int):
        """Log step-level metrics to wandb with step/ prefix."""
        if wandb.run is not None:
            data = {"global_step": step}
            data.update({f"step/{k}": v for k, v in metrics.items()})
            wandb.log(data)

    def _get_model_type(self) -> str:
        """Return the model_type string for visualization inference.

        Subclasses can override this to specify 'sam', 'segformer', etc.
        """
        return "default"

    def _visualize_predictions(
        self,
        loader=None,
        vis_dir=None,
        epoch=None,
        predictions_cache: Dict = None,
    ):
        """Visualize predictions with a unified implementation.

        Supports three modes:
        1. loader provided -> run model inference on loader (validation).
        2. predictions_cache provided -> use pre-computed results (test).
        3. Neither -> iterate test_loaders with model inference (fallback).

        Subclasses can override _get_model_type() to customise inference.
        """
        from utils.visualize import visualize_predictions, visualize_from_predictions

        num_vis_samples = self.cfg.get("visualization", {}).get("num_samples", 10)
        model_type = self._get_model_type()
        img_size = getattr(self, "img_size", None)

        # Mode 1: validation visualization
        if loader is not None:
            visualize_predictions(
                self.model,
                loader,
                self.device,
                self.cfg.data.num_classes,
                vis_dir,
                num_samples=num_vis_samples,
                model_type=model_type,
                img_size=img_size,
                phase_name="val",
            )
            return

        # Mode 2/3: test visualization
        if vis_dir is None:
            vis_dir = self.exp_dir / "visualizations" / f"epoch_{self.current_epoch + 1}"

        sample_msg = "all" if num_vis_samples is None else f"{num_vis_samples}"
        self.logger.info(
            f"Generating {sample_msg} visualizations for epoch {self.current_epoch + 1}..."
        )

        for name, test_loader in self._iter_test_loaders():
            save_dir = vis_dir / name if isinstance(self.test_loader, dict) else vis_dir
            phase_name = f"test_{name}" if isinstance(self.test_loader, dict) else "test"

            if predictions_cache and name in predictions_cache:
                cached = predictions_cache[name]
                images_list, preds_list, masks_list = cached[0], cached[1], cached[2]
                filenames_list = cached[3] if len(cached) > 3 else None
                visualize_from_predictions(
                    images_list,
                    preds_list,
                    masks_list,
                    self.cfg.data.num_classes,
                    save_dir,
                    num_samples=num_vis_samples,
                    phase_name=phase_name,
                    filenames_list=filenames_list,
                )
            else:
                visualize_predictions(
                    self.model,
                    test_loader,
                    self.device,
                    self.cfg.data.num_classes,
                    save_dir,
                    num_samples=num_vis_samples,
                    model_type=model_type,
                    img_size=img_size,
                    phase_name=phase_name,
                )

        self.logger.info(f"Visualizations saved to {vis_dir}")

    def _visualize_validation(self, epoch: int):
        """Visualize validation predictions."""
        vis_dir = self.exp_dir / "visualizations" / f"epoch_{epoch + 1}_val"
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"Generating validation visualizations for epoch {epoch + 1}..."
        )
        self._visualize_predictions(
            loader=self.val_loader, vis_dir=vis_dir, epoch=epoch
        )

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training")

        num_epochs = self.cfg.training.get("num_epochs", 100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Visualize validation every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._visualize_validation(epoch)

            # Log metrics (removed test_metrics from per-epoch loop)
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if self.early_stopping is not None:
                self.early_stopping(
                    val_metrics.get("Dice", val_metrics.get("dice", 0.0))
                )

                if self.early_stopping.should_stop():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Test with best model
        if self.best_model_path is not None:
            self.logger.info("Testing with best model")

            self._load_checkpoint(self.best_model_path)
            test_metrics = self.test()
            self._save_test_results(test_metrics)

        # Capture run ID for pipeline integration before potentially finishing
        self.wandb_run_id = (
            self.wandb_run.id
            if hasattr(self, "wandb_run") and self.wandb_run is not None
            else None
        )

        # In pipeline mode, keep the wandb run open so the distillation stage
        # can continue logging to the same run. Otherwise, finish normally.
        in_pipeline = self.cfg.get("pipeline", {}).get("enabled", False)
        if not in_pipeline:
            wandb.finish()

        self.logger.info("Training completed!")

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict = None,
    ):
        """Log training and validation metrics."""
        num_epochs = self.cfg.training.get("num_epochs", 100)
        train_items = {k: v for k, v in train_metrics.items() if isinstance(v, (float, int))}
        val_items = {k: v for k, v in val_metrics.items() if isinstance(v, (float, int))}

        # Log to console
        self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        self.logger.info(
            "Train:\n    " + ", ".join(f"{k}: {v:.4f}" for k, v in train_items.items())
        )
        self.logger.info(
            "Val:\n    " + ", ".join(f"{k}: {v:.4f}" for k, v in val_items.items())
        )
        if test_metrics is not None:
            test_items = {k: v for k, v in test_metrics.items() if isinstance(v, (float, int))}
            self.logger.info(
                "Test:\n    " + ", ".join(f"{k}: {v:.4f}" for k, v in test_items.items())
            )

        # Log to wandb (epoch-level metrics)
        wandb_metrics = {"epoch": epoch + 1}
        wandb_metrics.update({f"train/{k}": v for k, v in train_items.items()})
        wandb_metrics.update({f"val/{k}": v for k, v in val_items.items()})
        if test_metrics is not None:
            wandb_metrics.update({f"test/{k}": v for k, v in test_items.items()})
        if wandb.run is not None:
            wandb.log(wandb_metrics)

    def _save_checkpoint(self, epoch: int, val_metrics: Dict):
        """Save model checkpoint."""
        dice_score = val_metrics.get("Dice", val_metrics.get("dice", 0.0))

        # Save best model
        if dice_score > self.best_metric:
            self.best_metric = dice_score
            self.best_model_path = (
                self.ckpt_dir / f"best_epoch_{epoch + 1}_dice{dice_score:.4f}.pth"
            )
            self._save_model(self.best_model_path)
            self.logger.info(f"Saved best model: {self.best_model_path}")

        # Periodic checkpoint
        save_interval = self.cfg.get("training", {}).get("save_interval", 20)
        if (epoch + 1) % save_interval == 0:
            ckpt_path = self.ckpt_dir / f"epoch_{epoch + 1}.pth"
            self._save_model(ckpt_path)
            self.logger.info(f"Saved checkpoint: {ckpt_path}")

    def _save_model(self, path: Path):
        """Save model to path. Can be overridden by subclasses for custom saving."""
        torch.save(self.model.state_dict(), str(path))

    def _load_checkpoint(self, path: Path):
        """Load model from checkpoint. Can be overridden by subclasses for custom loading."""
        self.logger.info(f"Loading checkpoint: {path}")
        self.model.load_state_dict(torch.load(str(path)))

    def _save_test_results(self, test_metrics: Dict):
        """Save test results to file and log to wandb as final_test/."""
        test_results_path = self.exp_dir / "test_results.txt"
        with open(test_results_path, "w") as f:
            f.write("Test Results\n")
            f.write(f"Best Model: {self.best_model_path.name}\n")
            for key, value in test_metrics.items():
                if isinstance(value, (float, int)):
                    f.write(f"{key}: {value:.4f}\n")

        self.logger.info(f"Test results saved to {test_results_path}")

        # Log to wandb with final_test/ prefix
        if wandb.run is not None:
            final_metrics = {
                f"final_test/{k}": v
                for k, v in test_metrics.items()
                if isinstance(v, (float, int))
            }
            wandb.log(final_metrics)
            wandb.run.summary.update(final_metrics)

    def run_test_only(self, checkpoint_path: str):
        """Run test-only mode with a specific checkpoint."""
        self.logger.info("TEST-ONLY MODE")
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._load_checkpoint(checkpoint_path)

        # Initialize wandb for test-only run
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        self.wandb_run = wandb.init(
            entity=self.cfg.get("wandb", {}).get("entity", "hheo"),
            project=self.cfg.get("wandb", {}).get("project", "TinyUSFM"),
            name=f"{self.cfg.model.name}_test_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self._build_wandb_config(is_sweep=is_sweep),
            tags=["test-only"],
            mode=(
                "disabled"
                if self.cfg.get("wandb", {}).get("disabled", False)
                else self.cfg.get("wandb", {}).get("mode", None)
            ),
        )

        # Run test
        test_metrics = self.test()

        # Save results
        self.best_model_path = checkpoint_path
        self._save_test_results(test_metrics)

        wandb.finish()
        self.logger.info("Test-only evaluation completed!")
