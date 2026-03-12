"""
TinyUSFM Trainer

This module provides a trainer for TinyUSFM segmentation models.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from hydra.utils import instantiate

from .base_trainer import BaseTrainer
from utils.data_processing_seg import SegDatasetProcessor
from utils.load_model_seg import load_model_seg
from omegaconf import OmegaConf
from utils.schedule import build_scheduler, get_lr_decay_param_groups


def _build_criterion(num_classes: int) -> nn.Module:
    """Build loss function based on number of classes."""
    return nn.BCEWithLogitsLoss()


def _compute_loss(
    criterion: nn.Module, outputs: torch.Tensor, masks: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute loss with proper mask formatting."""
    return criterion(outputs, masks)


def _build_optimizer(model: nn.Module, cfg) -> optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = cfg.get("optimizer", {})
    if "_target_" in opt_cfg:
        opt_cfg = OmegaConf.to_container(opt_cfg, resolve=True)

        # Handle layer decay
        use_layer_decay = opt_cfg.pop("use_layer_decay", False)
        # Handle gradient clipping
        opt_cfg.pop("gradient_clip", None)

        if use_layer_decay:
            base_lr = opt_cfg.get("lr", cfg.training.get("lr", 0.0001))
            weight_decay = opt_cfg.get("weight_decay", 0.0)
            num_layers = opt_cfg.pop("num_layers", 12)
            layer_decay = opt_cfg.pop("layer_decay", 0.8)

            param_groups = get_lr_decay_param_groups(
                model=model,
                base_lr=base_lr,
                weight_decay=weight_decay,
                num_layers=num_layers,
                layer_decay=layer_decay,
            )
            # Use instantiate to create the optimizer with the param_groups
            return instantiate(
                {"_target_": opt_cfg["_target_"], **opt_cfg}, param_groups
            )
        else:
            # Normal instantiation
            return instantiate(opt_cfg, model.parameters())

    base_lr = cfg.training.get("lr", 0.0001)
    weight_decay = opt_cfg.get("weight_decay", 0)

    param_groups = get_lr_decay_param_groups(
        model=model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        num_layers=12,
        layer_decay=0.8,
    )
    return optim.AdamW(param_groups)


class TinyUSFMTrainer(BaseTrainer):
    """Trainer for TinyUSFM models."""

    def __init__(self, cfg):
        """Initialize TinyUSFM trainer."""
        super().__init__(cfg)
        self.criterion: Optional[nn.Module] = None

    def _create_model(self):
        """Create or use provided TinyUSFM model."""
        if self.model is None:
            self.model = instantiate(self.cfg.model)

        self.model.load_checkpoint()
        self.model = self.model.to(self.device)

        # Handle head-only training
        train_head_only = self.cfg.training.get("train_head_only", False)
        if train_head_only:
            self.logger.info("Head-only training enabled. Freezing backbone...")
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Log model info
        self._log_model_info()

        self.criterion = _build_criterion(self.cfg.data.num_classes)

    def _create_dataloaders(self):
        """Create data loaders."""
        self.train_loader, self.val_loader, self.test_loader = (
            SegDatasetProcessor.build_data_loaders(self.cfg)
        )
        self._log_data_sizes()

    def _create_optimizer(self):
        """Create optimizer."""
        self.optimizer = _build_optimizer(self.model, self.cfg)
        lr = self.cfg.training.get("lr", 0.0001)
        self.logger.info(f"Optimizer: AdamW, LR: {lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        use_reduce_on_plateau = self.cfg.get("scheduler", {}).get(
            "use_reduce_on_plateau", False
        )

        if use_reduce_on_plateau:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.cfg.scheduler.get("factor", 0.5),
                patience=self.cfg.scheduler.get("patience", 5),
                min_lr=self.cfg.scheduler.get("min_lr", 1e-7),
            )
            self.logger.info(f"Using ReduceLROnPlateau scheduler")
        else:
            self.scheduler = build_scheduler(self.optimizer, self.cfg)
            self.logger.info("Using WarmupPolyLR scheduler")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        # Handle warmup
        warmup_epochs = self.cfg.training.get("warmup_epochs", 0)
        if epoch < warmup_epochs:
            lr = self.cfg.training.lr * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        self.model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs}",
        )

        for images, masks, *_ in train_pbar:
            images, masks = images.to(self.device), masks.to(self.device).float()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            loss = _compute_loss(
                self.criterion, outputs, masks, self.cfg.data.num_classes
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            max_norm = self.cfg.optimizer.get("gradient_clip", {}).get(
                "max_norm", 1.0
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_norm
            )

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate epoch loss
        epoch_loss = running_loss / len(self.train_loader.dataset)
        current_lr = self.optimizer.param_groups[0]["lr"]

        metrics = {"loss": epoch_loss, "lr": current_lr}

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        val_loss = 0.0

        val_pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs} [Val]",
        )

        with torch.no_grad():
            for images, masks, *_ in val_pbar:
                images, masks = images.to(self.device), masks.to(self.device).float()

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss = _compute_loss(
                    self.criterion, outputs, masks, self.cfg.data.num_classes
                )

                val_loss += loss.item() * images.size(0)

        val_loss /= len(self.val_loader.dataset)

        # Evaluate metrics
        val_metrics = self.evaluator.evaluate_model(
            self.model, self.val_loader, self.device, self.cfg.data.num_classes
        )

        # Update scheduler
        use_reduce_on_plateau = self.cfg.get("scheduler", {}).get(
            "use_reduce_on_plateau", False
        )
        if use_reduce_on_plateau and self.scheduler is not None:
            self.scheduler.step(val_metrics["Dice"])
        elif (
            not use_reduce_on_plateau
            and self.scheduler is not None
            and epoch >= self.cfg.training.get("warmup_epochs", 0)
        ):
            self.scheduler.step()

        # Add loss to metrics
        val_metrics["loss"] = val_loss

        self.evaluator.print_metrics(val_metrics, phase="validation")

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        test_metrics = {}
        predictions_cache = {}
        for name, loader in self._iter_test_loaders():
            if isinstance(self.test_loader, dict):
                self.logger.info(f"Testing on dataset: {name}")

            results = self.evaluator.evaluate_model(
                self.model,
                loader,
                self.device,
                self.cfg.data.num_classes,
                return_predictions=True,
            )
            metrics, images_list, preds_list, masks_list, filenames_list = results
            predictions_cache[name] = (images_list, preds_list, masks_list, filenames_list)

            self.evaluator.print_metrics(
                metrics,
                phase=f"test_{name}" if isinstance(self.test_loader, dict) else "test",
            )

            if isinstance(self.test_loader, dict):
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
            else:
                test_metrics = metrics

        # Visualize predictions
        vis_dir = self.exp_dir / "visualizations" / "test"
        self._visualize_predictions(
            predictions_cache=predictions_cache, vis_dir=vis_dir
        )

        return test_metrics

    def _visualize_predictions(
        self,
        loader=None,
        vis_dir=None,
        epoch=None,
        predictions_cache: Dict[
            str, Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
        ] = None,
    ) -> None:
        """Visualize predictions. Delegates to base class implementation."""
        super()._visualize_predictions(
            loader=loader,
            vis_dir=vis_dir,
            epoch=epoch,
            predictions_cache=predictions_cache,
        )


try:
    import lightning as L
except ImportError:
    L = None


class TinyUSFMLightningModule(L.LightningModule if L is not None else object):
    """Lightning module for TinyUSFM training."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = load_model_seg(self.cfg, torch.device("cpu"))
        self.criterion = _build_criterion(self.cfg.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute loss for current batch."""
        return _compute_loss(self.criterion, outputs, masks, self.cfg.model.num_classes)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, masks, *_ = batch
        outputs = self.model(images)
        loss = self._compute_loss(outputs, masks)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List]:
        optimizer = _build_optimizer(self.model, self.cfg)
        scheduler = build_scheduler(optimizer, self.cfg)
        return [optimizer], [scheduler]

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, masks, *_ = batch
        outputs = self.model(images)
        loss = self._compute_loss(outputs, masks)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return loss

    def test_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        images, masks, *extra = batch
        filenames = list(extra[-1]) if extra and isinstance(extra[-1], (list, tuple)) and extra[-1] and isinstance(extra[-1][0], str) else None
        outputs = self.model(images)
        loss = self._compute_loss(outputs, masks)

        preds = (
            torch.sigmoid(outputs)
            if self.cfg.model.num_classes == 1
            else torch.softmax(outputs, dim=1)
        )

        vis_dir = Path(self.trainer.logger.log_dir) / "visualizations"
        self.visualize_predictions_batch(
            images.cpu(),
            preds.cpu(),
            masks.cpu(),
            vis_dir,
            batch_idx=batch_idx,
            num_samples=10,
            filenames=filenames,
        )

        self.log(
            "test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return loss

    def configure_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        gradient_clip_val = self.cfg.optimizer.get("gradient_clip", {}).get(
            "max_norm", 1.0
        )
        gradient_clip_algorithm = "norm"
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        images, *_ = batch
        return self.model(images)

    def visualize_predictions_batch(
        self,
        images: torch.Tensor,
        preds: torch.Tensor,
        masks: torch.Tensor,
        vis_dir: Path,
        batch_idx: int = 0,
        num_samples: int = 10,
        filenames: list = None,
    ) -> None:
        from utils.visualize import _visualize_sample_arrays

        vis_dir.mkdir(parents=True, exist_ok=True)

        images_np = images.numpy()
        preds_np = preds.numpy()
        masks_np = masks.numpy()

        sample_count = batch_idx * images_np.shape[0]

        for i in range(images_np.shape[0]):
            fname = filenames[i] if filenames else None
            file_label = fname or f"{sample_count:03d}"
            save_path = vis_dir / f"sample_{file_label}.png"
            _visualize_sample_arrays(
                images_np[i], masks_np[i], preds_np[i],
                save_path, self.cfg.model.num_classes,
                filename=fname,
            )

            sample_count += 1
            if sample_count >= num_samples:
                break
