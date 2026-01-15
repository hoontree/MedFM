"""
TinyUSFM Trainer

This module provides a trainer for TinyUSFM segmentation models.
"""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .base_trainer import BaseTrainer
from utils.data_processing_seg import SegDatasetProcessor
from utils.load_model_seg import load_model_seg
from utils.schedule import build_scheduler, get_lr_decay_param_groups


class TinyUSFMTrainer(BaseTrainer):
    """Trainer for TinyUSFM models."""

    def __init__(self, cfg):
        """Initialize TinyUSFM trainer."""
        super().__init__(cfg)

        # TinyUSFM-specific attributes
        self.criterion = None

    def _create_model(self):
        """Create TinyUSFM model."""
        self.model = load_model_seg(self.cfg, self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Setup loss function
        if self.cfg.model.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _create_dataloaders(self):
        """Create data loaders."""
        self.train_loader, self.val_loader, self.test_loader = SegDatasetProcessor.build_data_loaders(self.cfg)
        self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")
        if isinstance(self.test_loader, dict):
            total_test = sum(len(loader.dataset) for loader in self.test_loader.values())
            self.logger.info(f"Test set size (Total): {total_test}")
            for name, loader in self.test_loader.items():
                self.logger.info(f"  - {name}: {len(loader.dataset)}")
        else:
            self.logger.info(f"Test set size: {len(self.test_loader.dataset)}")

    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_name = self.cfg.optimizer.get('name', 'Adam')
        lr = self.cfg.training.get('lr', 0.0001)
        weight_decay = self.cfg.optimizer.get('weight_decay', 0)

        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            # Use layer-wise learning rate decay if available
            param_groups = get_lr_decay_param_groups(
                model=self.model,
                base_lr=lr,
                weight_decay=weight_decay,
                num_layers=12,
                layer_decay=0.8
            )
            self.optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.logger.info(f"Optimizer: {optimizer_name}, LR: {lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        use_reduce_on_plateau = self.cfg.get('scheduler', {}).get('use_reduce_on_plateau', False)

        if use_reduce_on_plateau:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.cfg.scheduler.get('factor', 0.5),
                patience=self.cfg.scheduler.get('patience', 5),
                min_lr=self.cfg.scheduler.get('min_lr', 1e-7),
                verbose=True
            )
            self.logger.info(f"Using ReduceLROnPlateau scheduler")
        else:
            self.scheduler = build_scheduler(self.optimizer, self.cfg)
            self.logger.info("Using WarmupPolyLR scheduler")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        # Handle warmup
        warmup_epochs = self.cfg.training.get('warmup_epochs', 0)
        if epoch < warmup_epochs:
            lr = self.cfg.training.lr * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs} [Training]"
        )

        for images, masks, low_res_labels in train_pbar:
            images, masks = images.to(self.device), masks.to(self.device).float()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            if self.cfg.model.num_classes == 2:
                loss = self.criterion(outputs, masks.unsqueeze(1))
            else:
                loss = self.criterion(outputs, masks.long())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            gradient_clip_enabled = self.cfg.optimizer.get('grad_clip', {}).get('enabled', False)
            if gradient_clip_enabled:
                max_norm = self.cfg.optimizer.grad_clip.get('max_norm', 1.0)
                norm_type = self.cfg.optimizer.grad_clip.get('norm_type', 2)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=norm_type)

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate epoch loss
        epoch_loss = running_loss / len(self.train_loader.dataset)
        current_lr = self.optimizer.param_groups[0]['lr']

        metrics = {
            'loss': epoch_loss,
            'lr': current_lr
        }

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        val_loss = 0.0

        val_pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs} [Validation]"
        )

        with torch.no_grad():
            for images, masks, _ in val_pbar:
                images, masks = images.to(self.device), masks.to(self.device).float()

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                if self.cfg.model.num_classes == 2:
                    loss = self.criterion(outputs, masks.unsqueeze(1))
                else:
                    loss = self.criterion(outputs, masks.long())

                val_loss += loss.item() * images.size(0)

        val_loss /= len(self.val_loader.dataset)

        # Evaluate metrics
        val_metrics = self.evaluator.evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            self.cfg.model.num_classes
        )

        # Update scheduler
        use_reduce_on_plateau = self.cfg.get('scheduler', {}).get('use_reduce_on_plateau', False)
        if use_reduce_on_plateau and self.scheduler is not None:
            self.scheduler.step(val_metrics['Dice'])
        elif not use_reduce_on_plateau and self.scheduler is not None and epoch >= self.cfg.training.get('warmup_epochs', 0):
            self.scheduler.step()

        # Add loss to metrics
        val_metrics['loss'] = val_loss

        self.evaluator.print_metrics(val_metrics, phase='validation')

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        test_metrics = {}

        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                self.logger.info(f"Testing on dataset: {name}")
                metrics = self.evaluator.evaluate_model(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.model.num_classes
                )
                self.evaluator.print_metrics(metrics, phase=f'test_{name}')
                
                # Prefix metrics with dataset name
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
        else:
            test_metrics = self.evaluator.evaluate_model(
                self.model,
                self.test_loader,
                self.device,
                self.cfg.model.num_classes
            )
            self.evaluator.print_metrics(test_metrics, phase='test')

        # Visualize predictions
        self._visualize_predictions()

        return test_metrics

    def _visualize_predictions(self):
        """Visualize test predictions."""
        from utils.visualize import visualize_predictions

        vis_dir = self.exp_dir / "visualizations"
        num_vis_samples = self.cfg.get('visualization', {}).get('num_samples', 10)

        self.logger.info(f"Generating {num_vis_samples} visualizations...")

        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                start_dir = vis_dir / name
                visualize_predictions(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.model.num_classes,
                    start_dir,
                    num_vis_samples,
                    model_type='default',
                    phase_name=f"test_{name}"
                )
        else:
            visualize_predictions(
                self.model,
                self.test_loader,
                self.device,
                self.cfg.model.num_classes,
                vis_dir,
                num_vis_samples,
                model_type='default',
                phase_name="test"
            )

        self.logger.info(f"Visualizations saved to {vis_dir}")
