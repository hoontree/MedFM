"""
TinyUSFM Trainer

This module provides a trainer for TinyUSFM segmentation models.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate

from .base_trainer import BaseTrainer
from utils.load_model_seg import load_model_seg
from omegaconf import OmegaConf
from utils.schedule import build_scheduler, get_lr_decay_param_groups


def _build_criterion(num_classes: int) -> nn.Module:
    """Build loss function based on number of classes."""
    if num_classes == 1:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _masks_to_index(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert masks to the index format expected by CrossEntropyLoss.

    Handles two common mask formats:
      - one-hot  [B, C, H, W]  (float) → argmax over channel dim → [B, H, W] long
      - index    [B, 1, H, W] or [B, H, W]  (int/long)           → squeeze → [B, H, W] long
    """
    if masks.dim() == 4 and masks.shape[1] == num_classes:
        # one-hot encoded
        return masks.argmax(dim=1).long()
    # index map: squeeze channel dim if present
    return masks.squeeze(1).long()


def _compute_loss(
    criterion: nn.Module, outputs: torch.Tensor, masks: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute loss with proper mask formatting."""
    if num_classes == 1:
        return criterion(outputs, masks.float())
    return criterion(outputs, _masks_to_index(masks, num_classes))


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
            self.model = instantiate(self.cfg.model)  # load_checkpoint called inside __init__

        # Load pretrained SAM decoder weights for the sam_mask branch
        sam_ckpt = self.cfg.model.get("sam_decoder_checkpoint", None)
        if sam_ckpt and self.cfg.model.get("decoder_type", "fpn") == "sam_mask":
            self.logger.info(f"Loading SAM decoder weights from: {sam_ckpt}")
            self.model.load_sam_decoder_weights(sam_ckpt)

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
            self._sched_cadence = "plateau"
            self.logger.info(f"Using ReduceLROnPlateau scheduler")
        else:
            self.scheduler = build_scheduler(self.optimizer, self.cfg)
            self._sched_cadence = "epoch"
            self.logger.info("Using WarmupPolyLR scheduler")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """One epoch via the shared BaseTrainer._supervised_epoch loop, using the
        TinyUSFM recipe's per-batch compute — the SAME loop + compute a distillation
        task_only run of a TinyUSFM student uses, so the two are bit-identical.
        The epoch/plateau scheduler steps in BaseTrainer.train() (cadence-driven)."""
        if getattr(self, "_recipe", None) is None:
            from trainers.recipes import get_recipe
            self._recipe = get_recipe("tinyusfm", self.cfg)
        return self._supervised_epoch(
            self.model, self._recipe.compute_batch_loss,
            self.model.parameters, getattr(self, "_sched_cadence", "epoch"), epoch,
        )

    def validate(self, epoch: int, return_predictions: bool = False):
        """Validate model.

        When ``return_predictions=True``, also returns a predictions cache so that
        visualization reuses the same forward pass as metric computation.
        """
        self.model.eval()

        # Single forward-pass: compute metrics and loss together
        result = self.evaluator.evaluate_model(
            self.model, self.val_loader, self.device, self.cfg.data.num_classes,
            criterion=self.criterion,
            return_predictions=return_predictions,
        )

        if return_predictions:
            val_metrics, images_l, preds_l, masks_l, fnames_l, _ = result
        else:
            val_metrics = result

        # Scheduler stepping (epoch / plateau) now happens in BaseTrainer.train()
        # via _sched_cadence, so both normal and distill training step it identically.

        self.evaluator.print_metrics(val_metrics, phase="validation")

        if return_predictions:
            return val_metrics, {"__val__": (images_l, preds_l, masks_l, fnames_l)}
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
            metrics, images_list, preds_list, masks_list, filenames_list, per_sample = results
            predictions_cache[name] = (images_list, preds_list, masks_list, filenames_list, per_sample)

            self.evaluator.print_metrics(
                metrics,
                phase=f"test_{name}" if isinstance(self.test_loader, dict) else "test",
            )

            if isinstance(self.test_loader, dict):
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
            else:
                test_metrics = metrics

        # Visualize predictions (all samples)
        vis_dir = self.exp_dir / "visualizations" / f"epoch_{self.current_epoch + 1}"
        self._visualize_predictions(
            predictions_cache=predictions_cache, vis_dir=vis_dir, num_samples=10
        )

        return test_metrics, predictions_cache
