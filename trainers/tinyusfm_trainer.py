"""
TinyUSFM Trainer

This module provides a trainer for TinyUSFM segmentation models.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .base_trainer import BaseTrainer
from utils.data_processing_seg import SegDatasetProcessor
from utils.load_model_seg import load_model_seg
from utils.schedule import build_scheduler, get_lr_decay_param_groups


def _build_criterion(num_classes: int) -> nn.Module:
    """Build loss function based on number of classes."""
    if num_classes == 2:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _compute_loss(criterion: nn.Module, outputs: torch.Tensor, masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute loss with proper mask formatting."""
    if num_classes == 2:
        return criterion(outputs, masks.unsqueeze(1))
    return criterion(outputs, masks.long())


def _build_optimizer(model: nn.Module, cfg) -> optim.Optimizer:
    """Build optimizer from config."""
    optimizer_name = cfg.optimizer.get('name', 'Adam')
    lr = cfg.training.get('lr', 0.0001)
    weight_decay = cfg.optimizer.get('weight_decay', 0)

    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        param_groups = get_lr_decay_param_groups(
            model=model,
            base_lr=lr,
            weight_decay=weight_decay,
            num_layers=12,
            layer_decay=0.8
        )
        return optim.AdamW(param_groups, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class TinyUSFMTrainer(BaseTrainer):
    """Trainer for TinyUSFM models."""

    def __init__(self, cfg):
        """Initialize TinyUSFM trainer."""
        super().__init__(cfg)
        self.criterion: Optional[nn.Module] = None

    def _create_model(self):
        """Create TinyUSFM model."""
        self.model = load_model_seg(self.cfg, self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        self.criterion = _build_criterion(self.cfg.model.num_classes)

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
        self.optimizer = _build_optimizer(self.model, self.cfg)
        optimizer_name = self.cfg.optimizer.get('name', 'Adam')
        lr = self.cfg.training.get('lr', 0.0001)
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
            loss = _compute_loss(self.criterion, outputs, masks, self.cfg.model.num_classes)

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
                loss = _compute_loss(self.criterion, outputs, masks, self.cfg.model.num_classes)

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
    
    def _iter_test_loaders(self):
        """Iterate over test loaders with names."""
        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                yield name, loader
        else:
            yield "test", self.test_loader

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        test_metrics = {}
        predictions_cache = {}
        for name, loader in self.test_loader.items():
            if isinstance(self.test_loader, dict):
                self.logger.info(f"Testing on dataset: {name}")
                
                results = self.evaluator.evaluate_model(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.model.num_classes,
                    return_predictions=True
                )
                metrics, images_list, preds_list, masks_list = results
                predictions_cache[name] = (images_list, preds_list, masks_list)
                self.evaluator.print_metrics(metrics, phase=f'test_{name}')
                
                if isinstance(self.test_loader, dict):
                    for k, v in metrics.items():
                        test_metrics[f"{name}/{k}"] = v
                else:
                    test_metrics = metrics
        # Visualize predictions
        self._visualize_predictions(predictions_cache)

        return test_metrics

    def _visualize_predictions(self, predictions_cache: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]) -> None:
        """Visualize test predictions."""
        from utils.visualize import visualize_from_predictions

        vis_dir = self.exp_dir / "visualizations"
        num_vis_samples = self.cfg.get('visualization', {}).get('num_samples', 10)
        
        sample_msg = "all" if num_vis_samples is None else f"{num_vis_samples} samples"

        self.logger.info(f"Generating {sample_msg} visualizations for epoch {self.current_epoch+1}...")

        for name, loader in self._iter_test_loaders():
            save_dir = vis_dir / name if isinstance(self.test_loader, dict) else vis_dir
            phase_name = f"test_{name}" if isinstance(self.test_loader, dict) else "test"

            if predictions_cache and name in predictions_cache:
                # Use cached predictions (no additional forward pass)
                images_list, preds_list, masks_list = predictions_cache[name]
                visualize_from_predictions(
                    images_list,
                    preds_list,
                    masks_list,
                    self.cfg.data.num_classes,
                    save_dir,
                    num_samples=num_vis_samples,
                    phase_name=phase_name
                )
            else:
                # Fallback to original method if no cache
                from utils.visualize import visualize_predictions
                visualize_predictions(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.data.num_classes,
                    save_dir,
                    num_samples=num_vis_samples,
                    model_type='sam',
                    img_size=self.img_size,
                    phase_name=phase_name
                )

        self.logger.info(f"Visualizations saved to {vis_dir}")
        
class TinyUSFMLightningModule(L.LightningModule):
    """Lightning module for TinyUSFM training."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = load_model_seg(self.cfg, torch.device('cpu'))
        self.criterion = _build_criterion(self.cfg.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute loss for current batch."""
        return _compute_loss(self.criterion, outputs, masks, self.cfg.model.num_classes)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, masks, _ = batch
        outputs = self.model(images)
        loss = self._compute_loss(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List]:
        optimizer = _build_optimizer(self.model, self.cfg)
        scheduler = build_scheduler(optimizer, self.cfg)
        return [optimizer], [scheduler]

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, masks, _ = batch
        outputs = self.model(images)
        loss = self._compute_loss(outputs, masks)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        images, masks, _ = batch
        outputs = self.model(images)
        loss = self._compute_loss(outputs, masks)

        preds = torch.sigmoid(outputs) if self.cfg.model.num_classes == 2 else torch.softmax(outputs, dim=1)

        vis_dir = Path(self.trainer.logger.log_dir) / "visualizations"
        self.visualize_predictions_batch(
            images.cpu(),
            preds.cpu(),
            masks.cpu(),
            vis_dir,
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
            batch_idx=batch_idx,
            num_samples=10
        )

        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None
    ) -> None:
        if self.cfg.optimizer.get('grad_clip', {}).get('enabled', False):
            gradient_clip_val = self.cfg.optimizer.grad_clip.get('max_norm', 1.0)
            gradient_clip_algorithm = 'norm'
        super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        images, _, _ = batch
        return self.model(images)

    def visualize_predictions_batch(
        self,
        images: torch.Tensor,
        preds: torch.Tensor,
        masks: torch.Tensor,
        vis_dir: Path,
        mean: np.ndarray,
        std: np.ndarray,
        batch_idx: int = 0,
        num_samples: int = 10
    ) -> None:
        vis_dir.mkdir(parents=True, exist_ok=True)

        images_np = images.numpy()
        preds_np = preds.numpy()
        masks_np = masks.numpy()

        sample_count = batch_idx * images_np.shape[0]

        for i in range(images_np.shape[0]):
            img = images_np[i]
            pred = preds_np[i]
            mask = masks_np[i]

            # Denormalize image
            if img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)
                img = std * img + mean
                img = np.clip(img, 0, 1)
            elif img.shape[0] == 1:  # Grayscale
                img = img[0]
                img = std[0] * img + mean[0]
                img = np.clip(img, 0, 1)

            # Fix mask/pred dimensions for plotting (squeeze channel if 1)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]

            # Create visualization figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Input Image
            axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0].set_title('Input Image', fontsize=14)
            axes[0].axis('off')

            # Ground Truth
            axes[1].imshow(mask, cmap='jet')
            axes[1].set_title('Ground Truth', fontsize=14)
            axes[1].axis('off')

            # Prediction
            axes[2].imshow(pred, cmap='jet')
            axes[2].set_title('Prediction', fontsize=14)
            axes[2].axis('off')

            # Overlay (Input + Prediction)
            axes[3].imshow(img, cmap='gray' if img.ndim == 2 else None)
            cmap = plt.get_cmap('jet')
            colored_pred = cmap(pred)
            colored_pred[..., 3] = 0.4  # Set alpha
            axes[3].imshow(colored_pred)
            axes[3].set_title('Overlay', fontsize=14)
            axes[3].axis('off')

            plt.tight_layout()
            save_path = vis_dir / f"sample_{sample_count:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            sample_count += 1
            if sample_count >= num_samples:
                break