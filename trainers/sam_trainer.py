"""
SAM (Segment Anything Model) Trainer

This module provides a trainer for SAM models with LoRA adaptation.
"""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from importlib import import_module

from .base_trainer import BaseTrainer
from model.segment_anything import sam_model_registry
from utils.data_processing_seg import SegDatasetProcessor
from utils.sam_utils import DiceLoss


class SAMTrainer(BaseTrainer):
    """Trainer for SAM models."""

    def __init__(self, cfg):
        """Initialize SAM trainer."""
        super().__init__(cfg)

        # SAM-specific attributes
        self.ce_loss = None
        self.bce_loss = None
        self.dice_loss = None
        self.multimask_output = False
        self.img_size = cfg.model.img_size

    def _create_model(self):
        """Create SAM model with LoRA."""
        # Register SAM model
        sam, img_embedding_size = sam_model_registry[self.cfg.model.name](
            image_size=self.cfg.model.img_size,
            num_classes=self.cfg.data.num_classes,
            checkpoint=self.cfg.model.ckpt,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
        )

        # Load LoRA module
        pkg = import_module(self.cfg.model.module)
        self.model = pkg.LoRA_Sam(sam, self.cfg.model.rank).cuda()

        # Load LoRA checkpoint if provided
        if self.cfg.model.get('lora_ckpt') is not None:
            self.model.load_lora_parameters(self.cfg.model.lora_ckpt)
            self.logger.info(f"Loaded LoRA checkpoint: {self.cfg.model.lora_ckpt}")

        # Setup DataParallel
        if self.cfg.hardware.get('n_gpu', 1) > 1:
            self.model = nn.DataParallel(self.model)

        # Setup losses
        self.ce_loss = CrossEntropyLoss()

        pos_weight = torch.tensor([5.0], device='cuda')
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.dice_loss = DiceLoss(self.cfg.data.num_classes)

        # Set multimask output
        self.multimask_output = self.cfg.data.num_classes > 2

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

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
        base_lr = self.cfg.training.base_lr
        warmup = self.cfg.training.get('warmup', False)
        warmup_period = self.cfg.training.get('warmup_period', 250)

        # Adjust learning rate for warmup
        b_lr = base_lr / warmup_period if warmup else base_lr

        optimizer_name = self.cfg.optimizer.get('name', 'SGD')

        if optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=b_lr,
                betas=(0.9, 0.999),
                weight_decay=self.cfg.optimizer.get('weight_decay', 0.1)
            )
        else:
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=b_lr,
                momentum=0.9,
                weight_decay=0.0001
            )

        self.logger.info(f"Optimizer: {optimizer_name}, Initial LR: {b_lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler (using manual update in train_epoch)."""
        # SAM uses custom polynomial decay scheduler updated in train_epoch
        pass

    def _update_learning_rate(self, iter_num: int):
        """Update learning rate with warmup and polynomial decay."""
        base_lr = self.cfg.training.base_lr
        warmup = self.cfg.training.get('warmup', False)
        warmup_period = self.cfg.training.get('warmup_period', 250)
        max_iterations = self.cfg.training.max_epochs * len(self.train_loader)

        if warmup and iter_num < warmup_period:
            # Warmup phase
            lr = base_lr * ((iter_num + 1) / warmup_period)
        else:
            # Polynomial decay phase
            if warmup:
                shift_iter = iter_num - warmup_period
                shift_iter = max(0, shift_iter)
            else:
                shift_iter = iter_num

            current_iter = min(shift_iter, max_iterations)
            lr = base_lr * (1.0 - current_iter / max_iterations) ** 0.9

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def _calc_loss(self, outputs, label_batch, low_res_label_batch):
        """Calculate loss based on number of classes."""
        dice_weight = self.cfg.training.get('dice_param', 0.8)

        if self.cfg.data.num_classes == 2:
            # Binary segmentation
            logits = outputs['masks']
            target = label_batch.unsqueeze(1).float()

            # Ensure target and logits have the same resolution
            if logits.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')

            target = (target > 0.5).float()

            loss_ce = self.bce_loss(logits, target)
            loss_dice = self.dice_loss(logits, target)
        else:
            # Multi-class segmentation
            low_res_logits = outputs['low_res_logits']
            loss_ce = self.ce_loss(low_res_logits, low_res_label_batch.long())
            loss_dice = self.dice_loss(low_res_logits, low_res_label_batch, softmax=True)

        loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice

        return loss, loss_ce, loss_dice

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0

        train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.cfg.training.max_epochs}')

        for batch_idx, (image_batch, label_batch, low_res_label_batch) in enumerate(train_pbar):
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()

            # Forward pass
            outputs = self.model(image_batch, False, self.img_size)

            # Calculate loss
            loss, loss_ce, loss_dice = self._calc_loss(outputs, label_batch, low_res_label_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip_config = self.cfg.get('optimizer', {}).get('grad_clip', {})
            if grad_clip_config.get('enabled', False):
                max_norm = grad_clip_config.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            self.optimizer.step()

            # Update learning rate
            lr = self._update_learning_rate(self.global_step)

            # Update metrics
            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_dice_loss += loss_dice.item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })

            # Log to tensorboard and wandb
            if self.global_step % 10 == 0:
                if self.writer is not None:
                    self.writer.add_scalar('info/lr', lr, self.global_step)
                    self.writer.add_scalar('info/total_loss', loss.item(), self.global_step)
                    self.writer.add_scalar('info/loss_ce', loss_ce.item(), self.global_step)
                    self.writer.add_scalar('info/loss_dice', loss_dice.item(), self.global_step)

                import wandb
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_ce': loss_ce.item(),
                    'train/loss_dice': loss_dice.item(),
                    'learning_rate': lr
                }, step=self.global_step)

            self.global_step += 1

        # Calculate average losses
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'loss_ce': total_ce_loss / num_batches,
            'loss_dice': total_dice_loss / num_batches,
        }

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        val_metrics = self.evaluator.evaluate_model_sam(
            self.model,
            self.val_loader,
            self.device,
            self.cfg.data.num_classes,
            img_size=self.img_size
        )

        self.evaluator.print_metrics(val_metrics, phase='validation')

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        test_metrics = {}

        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                self.logger.info(f"Testing on dataset: {name}")
                metrics = self.evaluator.evaluate_model_sam(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.data.num_classes,
                    img_size=self.img_size
                )
                self.evaluator.print_metrics(metrics, phase=f'test_{name}')
                
                # Prefix metrics
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
        else:
            test_metrics = self.evaluator.evaluate_model_sam(
                self.model,
                self.test_loader,
                self.device,
                self.cfg.data.num_classes,
                img_size=self.img_size
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
                    self.cfg.data.num_classes,
                    start_dir,
                    num_samples=num_vis_samples,
                    model_type='sam',
                    img_size=self.img_size,
                    phase_name=f"test_{name}"
                )
        else:
            visualize_predictions(
                self.model,
                self.test_loader,
                self.device,
                self.cfg.data.num_classes,
                vis_dir,
                num_samples=num_vis_samples,
                model_type='sam',
                img_size=self.img_size,
                phase_name="test"
            )

        self.logger.info(f"Visualizations saved to {vis_dir}")

    def _save_model(self, path: Path):
        """Save SAM model (LoRA parameters)."""
        try:
            self.model.save_lora_parameters(str(path))
        except AttributeError:
            # Handle DataParallel
            self.model.module.save_lora_parameters(str(path))

    def _load_checkpoint(self, path: Path):
        """Load SAM model checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        try:
            self.model.load_lora_parameters(str(path))
        except AttributeError:
            # Handle DataParallel
            self.model.module.load_lora_parameters(str(path))
