"""
Base Trainer for Multi-Model Training Framework

This module provides a base class for training different models with common functionalities.
"""

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

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
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

        if self.mode == 'max':
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

    def __init__(self, cfg: DictConfig):
        """
        Initialize base trainer.

        Args:
            cfg: Configuration object (OmegaConf DictConfig)
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize attributes
        self.model = None
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

    def setup(self, mode: str = 'train'):
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
        if mode == 'train':
            self._setup_wandb()

        # Create data loaders
        self._create_dataloaders()

        # Create model
        self._create_model()

        # Setup training components (only for training mode)
        if mode == 'train':
            self._create_optimizer()
            self._create_scheduler()
            self._setup_early_stopping()

        self.logger.info(f"Setup completed for {mode} mode")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.cfg.get('hardware', {}).get('seed', 1234)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        deterministic = self.cfg.get('hardware', {}).get('deterministic', True)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def _setup_directories(self, mode: str):
        """Setup experiment directories."""
        # Get model name
        model_name = self.cfg.model.get("adaptation_mode", self.cfg.model.name)
        if self.cfg.data.get('name') == 'dynamic':
            dataset_name = "+".join(self.cfg.data.train)
        else:
            dataset_name = self.cfg.data.name

        # Create base directory
        logs_root = Path(self.cfg.get('output', {}).get('dir', 'logs'))

        # Create experiment tags
        exp_tags = self._create_exp_tags()

        # Create timestamp-based experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir_name = timestamp + ("_" + "_".join(exp_tags) if exp_tags else "")

        # Final experiment directory
        self.exp_dir = logs_root / model_name / dataset_name / exp_dir_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir = self.exp_dir

        # Save config
        config_file = self.exp_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            OmegaConf.save(self.cfg, f)

    def _create_exp_tags(self) -> list:
        """Create experiment tags based on hyperparameters."""
        exp_tags = []

        # Add custom tags from config
        if hasattr(self.cfg.training, 'batch_size'):
            exp_tags.append(f"bs{self.cfg.training.batch_size}")

        if hasattr(self.cfg.training, 'base_lr'):
            if self.cfg.training.base_lr != 0.01:
                exp_tags.append(f"lr{self.cfg.training.base_lr}")
        elif hasattr(self.cfg.training, 'lr'):
            if self.cfg.training.lr != 0.01:
                exp_tags.append(f"lr{self.cfg.training.lr}")

        return exp_tags

    def _setup_logger(self):
        """Setup logger."""
        log_file = self.exp_dir / "train.log"
        self.logger = setup_logger(str(log_file))
        self.logger.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb_config = self.cfg.get('wandb', {})

        wandb.init(
            entity=wandb_config.get('entity', 'hheo'),
            project=wandb_config.get('project', 'TinyUSFM'),
            name=f"{self.cfg.model.get('adaptation_mode', self.cfg.model.get('name', 'model'))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(self.cfg, resolve=True),
            dir=str(self.exp_dir)
        )

    def _setup_early_stopping(self):
        """Setup early stopping."""
        early_stop_cfg = self.cfg.get('training', {}).get('early_stopping', {})

        if early_stop_cfg.get('enabled', False):
            patience = early_stop_cfg.get('patience', 15)
            min_delta = early_stop_cfg.get('min_delta', 0.001)
            self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')
            self.logger.info(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

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

    def _visualize_predictions(self):
        """Visualize test predictions. Should be implemented by subclasses."""
        self.logger.warning(f"{self.__class__.__name__} does not implement _visualize_predictions")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training")

        num_epochs = self.cfg.training.get('num_epochs', self.cfg.training.get('max_epochs', 100))

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)
            
            # Test
            test_metrics = self.test()

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics, test_metrics=test_metrics)

            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if self.early_stopping is not None:
                self.early_stopping(val_metrics.get('Dice', val_metrics.get('dice', 0.0)))

                if self.early_stopping.should_stop():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Test with best model
        if self.best_model_path is not None:
            self.logger.info("Testing with best model")

            self._load_checkpoint(self.best_model_path)
            test_metrics = self.test()
            self._save_test_results(test_metrics)

        # Cleanup
        wandb.finish()

        self.logger.info("Training completed!")

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, test_metrics: Dict = None):
        """Log training and validation metrics."""
        # Log to console
        self.logger.info(f"\nEpoch {epoch + 1}/{self.cfg.training.get('num_epochs', self.cfg.training.get('max_epochs', 100))}")
        self.logger.info(f"Train:")
        self.logger.info(f"    " + ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        self.logger.info(f"Val:")
        self.logger.info(f"    " + ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        if test_metrics is not None:
            self.logger.info(f"Test:")
            self.logger.info(f"    " + ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))

        # Log to wandb (epoch-level metrics)
        wandb_metrics = {'epoch': epoch + 1}
        wandb_metrics.update({f'epoch_train/{k}': v for k, v in train_metrics.items()})
        wandb_metrics.update({f'epoch_val/{k}': v for k, v in val_metrics.items()})
        if test_metrics is not None:
            wandb_metrics.update({f'epoch_test/{k}': v for k, v in test_metrics.items()})
        wandb.log(wandb_metrics)

    def _save_checkpoint(self, epoch: int, val_metrics: Dict):
        """Save model checkpoint."""
        dice_score = val_metrics.get('Dice', val_metrics.get('dice', 0.0))

        # Save best model
        if dice_score > self.best_metric:
            self.best_metric = dice_score
            self.best_model_path = self.ckpt_dir / f'best_epoch_{epoch + 1}_dice{dice_score:.4f}.pth'
            self._save_model(self.best_model_path)
            self.logger.info(f"Saved best model: {self.best_model_path}")

        # Periodic checkpoint
        save_interval = self.cfg.get('training', {}).get('save_interval', 20)
        if (epoch + 1) % save_interval == 0:
            ckpt_path = self.ckpt_dir / f'epoch_{epoch + 1}.pth'
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
        """Save test results to file."""
        test_results_path = self.exp_dir / "test_results.txt"
        with open(test_results_path, "w") as f:
            f.write("Test Results\n")
            f.write(f"Best Model: {self.best_model_path.name}\n")
            for key, value in test_metrics.items():
                f.write(f"{key}: {value:.4f}\n")

        self.logger.info(f"Test results saved to {test_results_path}")

        # Log to wandb
        wandb.log({f'test/{k}': v for k, v in test_metrics.items()})

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
        wandb.init(
            entity=self.cfg.get('wandb', {}).get('entity', 'hheo'),
            project=self.cfg.get('wandb', {}).get('project', 'TinyUSFM'),
            name=f"{self.cfg.model.name}_test_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(self.cfg, resolve=True),
            tags=['test-only']
        )

        # Run test
        test_metrics = self.test()

        # Save results
        self.best_model_path = checkpoint_path
        self._save_test_results(test_metrics)

        wandb.finish()
        self.logger.info("Test-only evaluation completed!")
