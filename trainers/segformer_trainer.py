"""
Segformer Trainer

This module provides a trainer for Segformer models.
"""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor
from .base_trainer import BaseTrainer
from utils.data_processing_seg import SegDatasetProcessor
from utils.sam_utils import DiceLoss


class SegformerTrainer(BaseTrainer):
    """Trainer for Segformer models."""

    def __init__(self, cfg):
        """Initialize Segformer trainer."""
        super().__init__(cfg)

        # Segformer-specific attributes
        self.bce_loss = None
        self.dice_loss = None
        self.image_processor = None

    def _create_model(self):
        """Create Segformer model."""
        # Configure Segformer model
        config = SegformerConfig.from_pretrained(
            pretrained_model_name_or_path="nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=1 if self.cfg.data.num_classes == 2 else self.cfg.data.num_classes
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            config=config,
            ignore_mismatched_sizes=True,
        ).cuda()

        # Setup image processor
        self.image_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )

        # Setup losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(self.cfg.data.num_classes)

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

        optimizer_name = self.cfg.optimizer.get('name', 'AdamW')

        if optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=b_lr,
                betas=(0.9, 0.999),
                weight_decay=self.cfg.optimizer.get('weight_decay', 0.1)
            )
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=b_lr,
                momentum=0.9,
                weight_decay=0.0001
            )
        else:
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=b_lr,
                weight_decay=self.cfg.optimizer.get('weight_decay', 0.01)
            )

        self.logger.info(f"Optimizer: {optimizer_name}, Initial LR: {b_lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler (using manual update in train_epoch)."""
        # Segformer uses custom polynomial decay scheduler updated in train_epoch
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

    def _calc_loss(self, outputs, label_batch):
        """Calculate loss based on number of classes."""
        dice_weight = self.cfg.training.get('dice_param', 0.8)

        if self.cfg.data.num_classes == 2:
            # Binary segmentation
            logits = outputs
            target = label_batch.float()

            # Ensure target has correct shape (B, 1, H, W)
            if target.dim() == 3:
                target = target.unsqueeze(1)

            # Ensure target and logits have the same resolution
            if logits.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')

            target = (target > 0.5).float()

            loss_ce = self.bce_loss(logits, target)
            loss_dice = self.dice_loss(logits, target)
        else:
            # Multi-class segmentation
            # For multi-class, outputs should be (B, num_classes, H, W)
            # and labels should be (B, H, W) with class indices
            loss_ce = F.cross_entropy(outputs, label_batch.long())
            loss_dice = self.dice_loss(outputs, label_batch, softmax=True)

        loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice

        return loss, loss_ce, loss_dice

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0

        train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.cfg.training.max_epochs}')

        for batch_idx, (image_batch, label_batch) in enumerate(train_pbar):
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()

            # Forward pass
            model_outputs = self.model(image_batch)

            # Get logits from model output
            logits = model_outputs.logits

            # Resize logits to match label size
            if logits.shape[-2:] != label_batch.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=label_batch.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Calculate loss
            loss, loss_ce, loss_dice = self._calc_loss(logits, label_batch)

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

            # Log to wandb (step-level metrics)
            if self.global_step % 10 == 0:
                wandb.log({
                    'step_train/loss': loss.item(),
                    'step_train/loss_ce': loss_ce.item(),
                    'step_train/loss_dice': loss_dice.item(),
                    'step_train/learning_rate': lr,
                    'global_step': self.global_step
                })

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

        # Use evaluator to compute metrics in a single pass (like SAM trainer)
        val_metrics = self._evaluate_segformer(self.val_loader)

        self.evaluator.print_metrics(val_metrics, phase='validation')

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        test_metrics = {}

        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                self.logger.info(f"Testing on dataset: {name}")
                metrics = self._evaluate_segformer(loader)
                self.evaluator.print_metrics(metrics, phase=f'test_{name}')
                
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
        else:
            test_metrics = self._evaluate_segformer(self.test_loader)
            self.evaluator.print_metrics(test_metrics, phase='test')

        # Visualize predictions if configured
        if self.cfg.get('visualization', {}).get('enabled', True):
            self._visualize_predictions()

        return test_metrics

    def _evaluate_segformer(self, dataloader):
        """Evaluate Segformer model on given dataloader."""
        from medpy.metric.binary import hd95, dc, recall

        self.model.eval()

        if self.cfg.data.num_classes == 2:
            return self._evaluate_binary_segformer(dataloader)
        else:
            return self._evaluate_multiclass_segformer(dataloader)

    def _evaluate_binary_segformer(self, dataloader):
        """Evaluate binary segmentation for Segformer."""
        from medpy.metric.binary import hd95, dc, recall
        import numpy as np

        dice_list = []
        hd95_list = []
        iou_list = []
        sensitivity_list = []
        specificity_list = []
        pixel_acc_list = []
        threshold = 0.5

        with torch.no_grad():
            for image_batch, label_batch in tqdm(dataloader, desc="Evaluating"):
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()

                # Forward pass
                model_outputs = self.model(image_batch)
                logits = model_outputs.logits

                # Resize logits to match label size
                if logits.shape[-2:] != label_batch.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=label_batch.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Get predictions
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()

                for pred, gt in zip(preds, label_batch):
                    pred_np = pred.squeeze().cpu().numpy().astype(bool)
                    gt_np = gt.squeeze().cpu().numpy().astype(bool)

                    dice = dc(pred_np, gt_np)
                    if pred_np.any() and gt_np.any():
                        hausdorff = hd95(pred_np, gt_np)
                    elif not pred_np.any() and not gt_np.any():
                        hausdorff = 0
                    else:
                        hausdorff = 224
                    iou = self._compute_jaccard(pred_np, gt_np)
                    sens = recall(pred_np, gt_np)
                    spec = self._compute_specificity(pred_np, gt_np)
                    pixel_acc = (pred_np == gt_np).sum() / gt_np.size

                    dice_list.append(dice)
                    hd95_list.append(hausdorff)
                    iou_list.append(iou)
                    sensitivity_list.append(sens)
                    specificity_list.append(spec)
                    pixel_acc_list.append(pixel_acc)

        if len(dice_list) == 0:
            return {
                'Dice': 0.0, 'Dice_std': 0.0,
                'HD95': 0.0, 'HD95_std': 0.0,
                'IoU': 0.0, 'IoU_std': 0.0,
                'Sensitivity': 0.0, 'Sensitivity_std': 0.0,
                'Specificity': 0.0, 'Specificity_std': 0.0,
                'PixelAcc': 0.0, 'PixelAcc_std': 0.0
            }

        metrics = {
            'Dice': np.mean(dice_list),
            'Dice_std': np.std(dice_list),
            'HD95': np.mean(hd95_list),
            'HD95_std': np.std(hd95_list),
            'IoU': np.mean(iou_list),
            'IoU_std': np.std(iou_list),
            'Sensitivity': np.mean(sensitivity_list),
            'Sensitivity_std': np.std(sensitivity_list),
            'Specificity': np.mean(specificity_list),
            'Specificity_std': np.std(specificity_list),
            'PixelAcc': np.mean(pixel_acc_list),
            'PixelAcc_std': np.std(pixel_acc_list)
        }

        return metrics

    def _evaluate_multiclass_segformer(self, dataloader):
        """Evaluate multi-class segmentation for Segformer."""
        from medpy.metric.binary import hd95
        import numpy as np

        num_classes = self.cfg.data.num_classes
        dice_per_class = [[] for _ in range(num_classes)]
        iou_per_class = [[] for _ in range(num_classes)]
        pixel_acc_list = []
        sensitivity_per_class = [[] for _ in range(num_classes)]
        specificity_per_class = [[] for _ in range(num_classes)]
        hd95_per_class = [[] for _ in range(num_classes)]

        with torch.no_grad():
            for image_batch, label_batch in tqdm(dataloader, desc="Evaluating"):
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()

                # Forward pass
                model_outputs = self.model(image_batch)
                logits = model_outputs.logits

                # Resize logits to match label size
                if logits.shape[-2:] != label_batch.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=label_batch.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                preds = torch.argmax(logits, dim=1)

                for pred, gt in zip(preds, label_batch):
                    pred_np = pred.cpu().numpy()
                    gt_np = gt.cpu().numpy()

                    pixel_acc = (pred_np == gt_np).mean()
                    pixel_acc_list.append(pixel_acc)

                    for class_id in range(num_classes):
                        pred_class = (pred_np == class_id)
                        gt_class = (gt_np == class_id)

                        if not (gt_class.any() or pred_class.any()):
                            dice = 1.0
                            iou = 1.0
                            sensitivity = np.nan
                            specificity = np.nan
                            hd = np.nan
                        else:
                            tp = np.logical_and(pred_class, gt_class).sum()
                            dice = 2 * tp / (pred_class.sum() + gt_class.sum() + 1e-8)
                            iou = self._compute_jaccard(pred_class, gt_class)
                            fn = np.logical_and(~pred_class, gt_class).sum()
                            if (tp + fn) > 0:
                                sensitivity = tp / (tp + fn + 1e-8)
                            else:
                                sensitivity = np.nan
                            tn = np.logical_and(~pred_class, ~gt_class).sum()
                            fp = np.logical_and(pred_class, ~gt_class).sum()
                            if (tn + fp) > 0:
                                specificity = tn / (tn + fp + 1e-8)
                            else:
                                specificity = np.nan
                            if gt_class.any() and pred_class.any():
                                hd = hd95(pred_class.astype(np.bool_), gt_class.astype(np.bool_))
                            else:
                                hd = 224

                        dice_per_class[class_id].append(dice)
                        iou_per_class[class_id].append(iou)
                        sensitivity_per_class[class_id].append(sensitivity)
                        specificity_per_class[class_id].append(specificity)
                        hd95_per_class[class_id].append(hd)

        foreground_ids = list(range(1, num_classes))

        metrics = {}
        metrics['PixelAcc'] = np.nanmean(pixel_acc_list)
        metrics['PixelAcc_std'] = np.nanstd(pixel_acc_list)
        metrics['Dice'] = np.nanmean([np.nanmean(dice_per_class[i]) for i in foreground_ids])
        metrics['Dice_std'] = np.nanstd([item for i in foreground_ids for item in dice_per_class[i]])
        metrics['IoU'] = np.nanmean([np.nanmean(iou_per_class[i]) for i in foreground_ids])
        metrics['IoU_std'] = np.nanstd([item for i in foreground_ids for item in iou_per_class[i]])
        metrics['Sensitivity'] = np.nanmean([np.nanmean(sensitivity_per_class[i]) for i in foreground_ids])
        metrics['Sensitivity_std'] = np.nanstd([item for i in foreground_ids for item in sensitivity_per_class[i]])
        metrics['Specificity'] = np.nanmean([np.nanmean(specificity_per_class[i]) for i in foreground_ids])
        metrics['Specificity_std'] = np.nanstd([item for i in foreground_ids for item in specificity_per_class[i]])
        metrics['HD95'] = np.nanmean([np.nanmean(hd95_per_class[i]) for i in foreground_ids])
        metrics['HD95_std'] = np.nanstd([item for i in foreground_ids for item in hd95_per_class[i]])

        return metrics

    @staticmethod
    def _compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute specificity metric."""
        tn = np.logical_and(pred == 0, gt == 0).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        return tn / (tn + fp + 1e-8)

    @staticmethod
    def _compute_jaccard(pred, gt):
        """Compute Jaccard/IoU metric."""
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union

    def _visualize_predictions(self):
        """Visualize test predictions."""
        from utils.visualize import visualize_predictions

        vis_dir = self.exp_dir / "visualizations"
        num_vis_samples = self.cfg.get('visualization', {}).get('num_samples', 10)

        self.logger.info(f"Generating {num_vis_samples} visualizations...")

        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                self.logger.info(f"Visualizing results for {name}...")
                visualize_predictions(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.data.num_classes,
                    vis_dir / name,
                    num_samples=num_vis_samples,
                    model_type='segformer',
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
                model_type='segformer',
                phase_name="test"
            )

        self.logger.info(f"Visualizations saved to {vis_dir}")
