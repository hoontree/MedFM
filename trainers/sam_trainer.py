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
import wandb

from hydra.utils import instantiate
from .base_trainer import BaseTrainer
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
        self.img_size = cfg.model.img_size
        self.step_log_interval = 10

        self.num_epochs = self.cfg.training.get("num_epochs", 100)
        self.base_lr = float(self.cfg.training.get("lr", 1e-4))
        self.dice_loss_weight = self.cfg.training.get("dice_loss_weight", 0.8)
        self.moe_loss_weight = float(self.cfg.get("training", {}).get("moe_loss_weight", 0.0))
        self.gradient_clip_max_norm = float(
            self.cfg.optimizer.get("gradient_clip", {}).get("max_norm", 1.0)
        )
        self.warmup_config = self.cfg.training.get(
            "warmup", {"enabled": False, "steps": 0}
        )

    def _create_model(self):
        """Create SAM model using ModelBuilder."""
        # Keep trainer runtime img_size aligned with any pre-dataloader sync.
        self.model = instantiate(self.cfg.model).to(self.device)

        # Setup DataParallel
        if len(self.cfg.get("hardware", {}).get("gpu_ids", [0])) > 1:
            self.model = nn.DataParallel(self.model)

        self._setup_loss_functions()

        self.step_log_interval = 10
        self._log_model_configuration()

    def _create_optimizer(self):
        """Create optimizer."""
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr,
            weight_decay=self.cfg.optimizer.get("weight_decay", 0.1),
        )

        self.logger.info(f"Optimizer: AdamW, Base LR: {self.base_lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and polynomial decay."""
        warmup_cfg = self.warmup_config
        warmup_steps = warmup_cfg["steps"]
        warmup_enabled = warmup_cfg["enabled"]

        total_iters = self.num_epochs * len(self.train_loader)
        power = float(self.cfg.get("scheduler", {}).get("power", 0.9))

        # Minimum learning rate ratio (prevents LR from reaching 0)
        min_lr = float(self.cfg.get("scheduler", {}).get("min_lr", 1e-6))
        min_lr_ratio = min_lr / self.base_lr

        def lr_lambda(current_step: int) -> float:
            """Calculate learning rate multiplier for given step."""
            # Warmup phase: linear increase
            if warmup_enabled and warmup_steps > 0 and current_step < warmup_steps:
                return (current_step + 1) / warmup_steps

            # Polynomial decay phase
            shift_iter = (
                current_step - warmup_steps
                if (warmup_enabled and warmup_steps > 0)
                else current_step
            )
            shift_iter = min(max(0, shift_iter), total_iters)  # Clamp to valid range
            decay = (1.0 - shift_iter / total_iters) ** power
            return max(min_lr_ratio, decay)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        self.logger.info(
            f"Scheduler: LambdaLR with warmup={warmup_enabled}, warmup_steps={warmup_steps}, "
            f"max_iterations={total_iters}, power={power}, min_lr={min_lr}"
        )

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]["lr"]

    def _log_model_configuration(self):
        """Log resolved SAM adaptation and pretrained configuration."""
        self._log_model_info()

        encoder_mode = getattr(self.model, "encoder_mode", "N/A")
        decoder_mode = getattr(self.model, "decoder_mode", "N/A")
        use_alignment = getattr(self.model, "use_alignment", False)

        self.logger.info("SAM Configuration:")
        self.logger.info(
            "encoder_mode=%s, decoder_mode=%s, alignment=%s",
            encoder_mode,
            decoder_mode,
            use_alignment,
        )

        if wandb.run is not None:
            wandb.config.update(
                {
                    "model.encoder_mode": encoder_mode,
                    "model.decoder_mode": decoder_mode,
                    "model.use_alignment": use_alignment,
                },
                allow_val_change=True,
            )

    def _setup_loss_functions(self):
        """Setup loss functions based on number of classes."""
        num_classes = self.cfg.data.num_classes

        if num_classes == 1:
            pos_weight = torch.tensor([5.0], device=self.device)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.ce_loss = CrossEntropyLoss()

        self.dice_loss = DiceLoss(num_classes)

    def _calc_loss(self, outputs, label_batch, low_res_label_batch):
        """Calculate loss using unified channel-based approach."""
        dice_weight = self.dice_loss_weight
        moe_loss_weight = self.moe_loss_weight
        num_classes = self.cfg.data.num_classes

        logits = outputs["low_res_logits"]
        target = low_res_label_batch

        # Ensure target and logits have the same resolution.
        # Expected: target is downsampled to match low-res logits.
        # Upsampling (target smaller than logits) should never happen in normal use.
        if logits.shape[-2:] != target.shape[-2:]:
            if target.shape[-2] < logits.shape[-2] or target.shape[-1] < logits.shape[-1]:
                import warnings
                warnings.warn(
                    f"_calc_loss: target {tuple(target.shape[-2:])} is smaller than "
                    f"logits {tuple(logits.shape[-2:])} — upsampling target, which may "
                    "silently degrade training. Check data pipeline resolution.",
                    stacklevel=2,
                )
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        if num_classes == 1:
            loss_ce = self.bce_loss(logits, target)
            loss_dice = self.dice_loss(logits, target)
        else:
            # target is one-hot [B, C, H, W] float; CE expects class index [B, H, W] long
            target_idx = target.argmax(dim=1).long()
            loss_ce = self.ce_loss(logits, target_idx)
            loss_dice = self.dice_loss(logits, target, softmax=True, sigmoid=False)

        loss_moe = outputs.get("moe_loss", torch.tensor(0.0, device=logits.device))
        if not torch.is_tensor(loss_moe):
            loss_moe = torch.tensor(float(loss_moe), device=logits.device)

        loss = (
            (1 - dice_weight) * loss_ce
            + dice_weight * loss_dice
            + moe_loss_weight * loss_moe
        )

        return loss, loss_ce, loss_dice, loss_moe

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        total_moe_loss = 0.0

        train_pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"
        )

        for image_batch, label_batch, low_res_label_batch, *_ in train_pbar:
            image_batch = image_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            low_res_label_batch = low_res_label_batch.to(self.device)

            # Forward pass: use multimask output when num_classes > 1
            multimask_output = self.cfg.data.num_classes > 1
            outputs = self.model(image_batch, multimask_output, self.img_size)

            # Calculate loss
            loss, loss_ce, loss_dice, loss_moe = self._calc_loss(
                outputs, label_batch, low_res_label_batch
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_max_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Get current learning rate
            lr = self._get_current_lr()

            # Update metrics
            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_dice_loss += loss_dice.item()
            total_moe_loss += loss_moe.item()

            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

            # Log to wandb (step-level metrics)
            if self.global_step % self.step_log_interval == 0:
                self._log_step_metrics(
                    {
                        "loss": loss.item(),
                        "loss_ce": loss_ce.item(),
                        "loss_dice": loss_dice.item(),
                        "loss_moe": loss_moe.item(),
                        "lr": lr,
                    },
                    self.global_step,
                )

            self.global_step += 1

        # Calculate average losses
        num_batches = len(self.train_loader)
        metrics = {
            "loss": total_loss / num_batches,
            "loss_ce": total_ce_loss / num_batches,
            "loss_dice": total_dice_loss / num_batches,
            "loss_moe": total_moe_loss / num_batches,
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
            img_size=self.img_size,
        )

        self.evaluator.print_metrics(val_metrics, phase="validation")

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test model and visualize predictions using cached inference results."""
        self.model.eval()

        test_metrics = {}
        # Store predictions for visualization (avoid redundant forward pass)
        predictions_cache = {}

        for name, loader in self._iter_test_loaders():
            if isinstance(self.test_loader, dict):
                self.logger.info(f"Testing on dataset: {name}")

            # Get metrics and predictions in one forward pass
            result = self.evaluator.evaluate_model_sam(
                self.model,
                loader,
                self.device,
                self.cfg.data.num_classes,
                img_size=self.img_size,
                return_predictions=True,
            )
            metrics, images_list, preds_list, masks_list, filenames_list = result

            self.evaluator.print_metrics(
                metrics,
                phase=f"test_{name}" if isinstance(self.test_loader, dict) else "test",
            )

            # Store metrics with appropriate keys
            if isinstance(self.test_loader, dict):
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
            else:
                test_metrics = metrics

            # Cache predictions for visualization
            predictions_cache[name] = (images_list, preds_list, masks_list, filenames_list)

        # Visualize predictions using cached results (all samples)
        vis_dir = self.exp_dir / "visualizations" / f"epoch_{self.current_epoch + 1}"
        self._visualize_predictions(
            predictions_cache=predictions_cache, vis_dir=vis_dir, num_samples=10
        )

        return test_metrics, predictions_cache

    def _get_base_model(self):
        """Return the underlying LoRA_Sam, unwrapping DataParallel if needed."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_model_type(self) -> str:
        """SAM model type for visualization inference."""
        return "sam"

    def _save_model(self, path: Path):
        """Save SAM model (LoRA parameters)."""
        self._get_base_model().save_lora_parameters(str(path))

    def _load_checkpoint(self, path: Path):
        """Load SAM model checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        self._get_base_model().load_lora_parameters(str(path))
